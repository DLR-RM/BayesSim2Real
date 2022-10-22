# copied from https://github.com/BlackHC/laaos
import dataclasses
import os
import enum
import pprint
import typing
from collections.abc import MutableMapping, MutableSequence, MutableSet
from datetime import datetime
from io import TextIOBase
from typing import Iterator
from typing import List
from typing import TypeVar

T = TypeVar("T")  # Any type.
KT = TypeVar("KT")  # Key type.
VT = TypeVar("VT")  # Value type.
T_co = TypeVar("T_co", covariant=True)  # Any type covariant containers.
V_co = TypeVar("V_co", covariant=True)  # Any type covariant containers.
VT_co = TypeVar("VT_co", covariant=True)  # Value type covariant containers.


def generate_time_id():
    now = datetime.now()
    id = now.strftime("%Y-%m-%d-%H%M%S")
    return "_" + id


def can_iter(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    else:
        return True


class TypeHandler:
    """
    TypeHandlers extend the behavior of Store to support additional types.
    """

    def supports(self, obj):
        """Whether this type handler supports handling `obj`."""
        return False

    def wrap(self, obj, wrap):
        """
        Wraps obj for being used within a Store.

        :param obj: object to be wrapped
        :param wrap: outer wrap function for wrapping of fields
        :return: wrapped object
        """
        return obj

    def repr(self, obj, repr, store):
        """
        `repr` of obj.

        :param obj: object to be serialized
        :param repr: outer repr function for `repr`-ing of fields
        :param store: Store object for reference
        :return: `repr` of obj (as a string)
        """
        return obj


class WeakEnumHandler(TypeHandler):
    """Requires expose_symbols on safe_load."""

    def supports(self, obj):
        return isinstance(obj, enum.Enum)

    def wrap(self, obj, wrap):
        return obj

    def repr(self, obj: enum.Enum, repr, store):
        return f"{obj.__class__.__qualname__}.{obj.name}"


class StrEnumHandler(TypeHandler):
    """Requires custom handling on safe_load."""

    def supports(self, obj):
        return isinstance(obj, enum.Enum)

    def wrap(self, obj, wrap):
        return obj

    def repr(self, obj: enum.Enum, repr, store):
        return repr(str(obj))


class ToReprHandler(TypeHandler):
    """Convert anything to repr. This is a catch-all."""

    def supports(self, obj):
        return True

    def wrap(self, obj, wrap):
        return obj

    def repr(self, obj: enum.Enum, _repr, store):
        return repr(obj)


class Dataclasses2DictHandler(TypeHandler):
    """Requires custom handling on safe_load."""

    def supports(self, obj):
        return dataclasses.is_dataclass(obj) and not isinstance(obj, type)

    def wrap(self, obj, wrap):
        return obj

    def repr(self, obj, repr, store):
        return repr({f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)})


class Function2StrHandler(TypeHandler):
    def supports(self, obj):
        return isinstance(obj, typing.Callable)

    def wrap(self, obj, wrap):
        return obj

    def repr(self, obj, repr, store):
        return repr(f"{obj.__module__}.{obj.__qualname__}")


DefaultTypeHandlers = [
    Dataclasses2DictHandler(),
    StrEnumHandler(),
    Function2StrHandler(),
    ToReprHandler(),
]


class Store:
    def __init__(
        self,
        log: TextIOBase,
        *,
        uri=None,
        initial_data=None,
        type_handlers=None,
        append_mode=False,
    ):
        """
        Create a Store.

        :param log: TextIOBase to write to.
        :param uri: Identifier for the Store (filename or similar)
        :param initial_data: Initial data for the store dict.
        :param type_handlers: List of type handlers that get called in order
        to wrap and serialize unknown types.
        """
        if initial_data is None:
            initial_data = {}
        if type_handlers is None:
            type_handlers = ()

        self._type_handlers: List[TypeHandler] = type_handlers
        self._log = log
        self._uri = uri

        wrapped_initial_data = self._wrap(initial_data)
        self._root = StoreRoot(self, wrapped_initial_data)
        StoreAccessible.link(self._root, "store")

        if not append_mode:
            if wrapped_initial_data:
                Store.write(
                    self,
                    f"store = (\n{pprint.pformat(wrapped_initial_data, width=160, compact=True)}\n)",
                )
            else:
                Store.write(self, "store = {}")

    @property
    def uri(self):
        return self._uri

    def close(self):
        self._log.close()

    def _wrap(self, obj):
        if isinstance(obj, (int, float, complex, str, type(None), bool)):
            pass
        elif isinstance(obj, StoreAccessible) and obj._accessor is None:
            pass
        elif isinstance(obj, (list, StoreList)):
            obj = StoreList(self, [self._wrap(value) for value in obj])
        elif isinstance(obj, (dict, StoreDict)):
            obj = StoreDict(
                self, {key: self._wrap(value) for key, value in obj.items()}
            )
        elif isinstance(obj, (set, StoreSet)):
            obj = StoreSet(self, {self._wrap(value) for value in obj})
        elif can_iter(obj):
            obj = StoreList(self, [self._wrap(value) for value in iter(obj)])
        else:
            for type_handler in self._type_handlers:
                if type_handler.supports(obj):
                    return type_handler.wrap(obj, self._wrap)
            raise KeyError(f"{type(obj)} not supported for LAAOS!")
        return obj

    def _repr(self, obj):
        if isinstance(obj, (int, float, complex, str, type(None), bool)):
            return repr(obj)
        elif isinstance(obj, (list, StoreList)):
            return "[" + ", ".join(self._repr(value) for value in obj) + "]"
        elif isinstance(obj, (dict, StoreDict)):
            return (
                "{"
                + ", ".join(
                    f"{self._repr(key)}: {self._repr(value)}"
                    for key, value in obj.items()
                )
                + "}"
            )
        elif isinstance(obj, (set, StoreSet)):
            return (
                "{" + ", ".join(self._repr(value) for value in obj) + "}"
                if obj
                else "set()"
            )
        else:
            for type_handler in self._type_handlers:
                if type_handler.supports(obj):
                    return type_handler.repr(obj, self._repr, self)
        return repr(obj)

    @staticmethod
    def write(store: "Store", text):
        store._log.write(text + "\n")
        store._log.flush()

    @staticmethod
    def wrap(store: "Store", obj):
        return store._wrap(obj)

    @staticmethod
    def repr(store: "Store", obj):
        return store._repr(obj)

    @property
    def root(self) -> "StoreRoot":
        return self._root

    def __repr__(self):
        return repr(self._root)


class StoreAccessible(object):
    def __init__(self, store: Store):
        self._store = store
        self._accessor = None

    @property
    def laaos_store(self):
        return self._store

    def _check_accessor(self):
        assert self._accessor is not None, (
            "You tried to mutate a store collection after it has been unlinked!\n\n"
            "This triggers an exception because it would be too hard to figure out how "
            "to rewrite this into something executable."
        )

    def _wrap(self, obj):
        return Store.wrap(self._store, obj)

    def _repr(self, obj):
        return Store.repr(self._store, obj)

    def _write(self, text):
        return Store.write(self._store, text)

    def _unlink(self):
        self._accessor = None

    def _link(self, accessor):
        self._accessor = accessor

    @staticmethod
    def unlink(obj):
        if isinstance(obj, StoreAccessible):
            obj._unlink()

    @staticmethod
    def link(obj, accessor):
        if isinstance(obj, StoreAccessible):
            obj._link(accessor)


class StoreDict(MutableMapping, StoreAccessible):
    def __init__(self, store: Store, initial_data):
        super().__init__(store)
        self._data = {}
        self._data.update(initial_data)

    def _unlink(self):
        super()._unlink()
        for value in self._data.values():
            StoreAccessible.unlink(value)

    def _link(self, accessor):
        super()._link(accessor)
        for key, value in self._data.items():
            StoreAccessible.link(value, f"{self._accessor}[{self._repr(key)}]")

    def __getitem__(self, key: KT) -> VT_co:
        return self._data[key]

    def __setitem__(self, key: KT, value: VT) -> None:
        self._check_accessor()

        old_value = self._data.get(key, None)
        if old_value is value:
            return

        StoreAccessible.unlink(old_value)

        value = self._wrap(value)
        self._data[key] = value
        self._write(f"{self._accessor}[{self._repr(key)}]={self._repr(value)}")

        StoreAccessible.link(value, f"{self._accessor}[{self._repr(key)}]")

    def __delitem__(self, key: KT) -> None:
        if key not in self._data:
            # Early out with the correct exception
            del self._data[key]

        self._check_accessor()
        StoreAccessible.unlink(self._data.get(key, None))

        del self._data[key]

        self._write(f"del {self._accessor}[{self._repr(key)}]")

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[T_co]:
        return iter(self._data)

    def __repr__(self) -> str:
        return self._repr(self._data)


class StoreRoot(StoreDict):
    def __init__(self, store: Store, initial_data):
        super().__init__(store, initial_data)

    def close(self):
        return self._store.close()

    @property
    def uri(self):
        return self._store.uri


class StoreList(MutableSequence, StoreAccessible):
    def __init__(self, store, seq: list):
        super().__init__(store)
        self._seq = list(seq)

    def _unlink(self):
        super()._unlink()
        for value in self._seq:
            StoreAccessible.unlink(value)

    def _link(self, accessor):
        super()._link(accessor)
        for key, value in enumerate(self._seq):
            StoreAccessible.link(value, f"{self._accessor}[{self._repr(key)}]")

    def clear(self) -> None:
        self._check_accessor()
        for value in self._seq:
            StoreAccessible.unlink(value)
        self._seq.clear()
        self._write(f"{self._accessor}.clear()")

    def insert(self, index: int, obj: T) -> None:
        self._check_accessor()

        obj = self._wrap(obj)
        self._seq.insert(index, obj)

        self._write(f"{self._accessor}.insert({self._repr(index)}, {self._repr(obj)})")

        for i in range(index, len(self._seq)):
            StoreAccessible.link(self._seq[i], f"{self._accessor}[{i}]")

    def append(self, obj: T) -> None:
        self._check_accessor()

        obj = self._wrap(obj)
        self._seq.append(obj)

        self._write(f"{self._accessor}.append({self._repr(obj)})")
        StoreAccessible.link(obj, f"{self._accessor}[{len(self._seq)-1}]")

    def __getitem__(self, key) -> T:
        return self._seq[key]

    def __setitem__(self, key, value) -> None:
        assert not isinstance(
            key, slice
        ), "Slices are not supported for lists in the store!"
        if not -len(self._seq) <= key < len(self._seq):
            # Early out with the correct exception
            self._seq[key] = value

        if key < 0:
            key += len(self._seq)

        self._check_accessor()

        old_value = self._seq[key]
        if old_value is value:
            return

        StoreAccessible.unlink(old_value)

        value = self._wrap(value)
        self._seq[key] = value

        self._write(f"{self._accessor}[{self._repr(key)}] = {self._repr(value)}")
        StoreAccessible.link(value, f"{self._accessor}[{self._repr(key)}]")

    def __delitem__(self, key) -> None:
        if not -len(self._seq) <= key < len(self._seq):
            # Early out with the correct exception
            del self._seq[key]

        if key < 0:
            key += len(self._seq)

        self._check_accessor()

        StoreAccessible.unlink(self._seq[key])
        del self._seq[key]
        Store.write(self._store, f"del {self._accessor}[{self._repr(key)}]")

        for i in range(key, len(self._seq)):
            StoreAccessible.link(self._seq[i], f"{self._accessor}[{i}]")

    def __len__(self) -> int:
        return len(self._seq)

    def __repr__(self) -> str:
        return self._repr(self._seq)

    def __eq__(self, other):
        if isinstance(other, StoreList):
            return self._seq == other._seq
        return self._seq == other


class StoreSet(MutableSet, StoreAccessible):
    def __init__(self, store: Store, initial_data):
        super().__init__(store)
        self._set = set(initial_data)

    def add(self, x: T) -> None:
        self._check_accessor()

        self._set.add(x)
        self._write(f"{self._accessor}.add({self._repr(x)})")

    def discard(self, x: T) -> None:
        self._check_accessor()

        self._set.discard(x)
        self._write(f"{self._accessor}.discard({self._repr(x)})")

    def __contains__(self, x: object) -> bool:
        return x in self._set

    def __len__(self) -> int:
        return len(self._set)

    def __iter__(self) -> Iterator[T_co]:
        return iter(self._set)

    def __repr__(self) -> str:
        return self._repr(self._set)


def ensure_dirs(filename):
    abs_path = os.path.abspath(filename)
    abs_dir = os.path.dirname(abs_path)
    os.makedirs(abs_dir, exist_ok=True)


def open_file_store(
    store_name="results",
    suffix=None,
    ext=".py",
    prefix="laaos/",
    *,
    truncate=False,
    initial_data=None,
    type_handlers=None,
    exposed_symbols=None,
    extra_mappings=None,
) -> StoreRoot:
    """
    Opens a file store. Either truncates any existing store in the same file, or otherwise loads an existing store to
    append data. `initial_data` can be used for otherwise empty store.

    Append pattern:
    ```
    # config = ...
    store = open_file_store(suffix="", initial_data=dict(config=config))
    if store["config"] != config:
        raise ValueError("Reopened store but config mismatch!")
    ````

    :param store_name: name of the store
    :param suffix: if None, use the current time
    :param ext: file extension
    :param prefix: prefix file path
    :param truncate: whether to truncate the store if the file exists already, otherwise load to append
    :param initial_data: intial data to use for a new store
    :param type_handlers: type handlers for the store
    :param exposed_symbols: exposed symbols for the store (see `load_safe_str`)
    :param extra_mappings: extra symbol mappings for the store (see `load_safe_str`)
    """
    if suffix is None:
        suffix = generate_time_id()

    filename = f"{prefix}{store_name}{suffix}{ext}"
    ensure_dirs(filename)
    log = open(filename, "a+t" if not truncate else "wt")

    if not truncate:
        log.seek(0)
        existing_code = log.read()
    else:
        existing_code = None

    if existing_code:
        existing_store = safe_load_str(
            existing_code,
            exposed_symbols=exposed_symbols,
            extra_mappings=extra_mappings,
        )
        append_mode = True
    else:
        existing_store = initial_data
        append_mode = False

    store = Store(
        log,
        uri=filename,
        initial_data=existing_store,
        type_handlers=type_handlers,
        append_mode=append_mode,
    )
    return store.root


def safe_load_str(code: str, exposed_symbols=None, extra_mappings=None):
    global_symbols = dict(__builtins__=dict(set=set))

    if exposed_symbols is not None:
        global_symbols.update({symbol.__name__: symbol for symbol in exposed_symbols})

    if extra_mappings is not None:
        global_symbols.update(extra_mappings)

    root = dict()
    exec(code, global_symbols, root)
    return root["store"]


def safe_load(path: str, exposed_symbols=None, extra_mappings=None):
    with open(path, "rt") as file:
        return safe_load_str(
            file.read(), exposed_symbols=exposed_symbols, extra_mappings=extra_mappings
        )


def compact(source_path: str, destination_path: str):
    source_store = safe_load(source_path)

    ensure_dirs(destination_path)
    destination = open(destination_path, "wt")
    destination_store = Store(destination, uri=source_path, initial_data=source_store)
    destination_store.close()


def new_set(obj, accessor) -> set:
    obj[accessor] = set()
    return obj[accessor]


def new_dict(obj, accessor) -> dict:
    obj[accessor] = {}
    return obj[accessor]


def new_list(obj, accessor) -> list:
    obj[accessor] = []
    return obj[accessor]


def put(obj, accessor, data: T) -> T:
    obj[accessor] = data
    return obj[accessor]
