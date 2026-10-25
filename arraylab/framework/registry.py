"""Ordered lazy command routing without imports of application or features."""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Iterable


Command = Callable[[list[str]], int]
CommandFactory = Callable[[], Command]


def _copy_args(argv):
    return list(argv)


@dataclass(frozen=True)
class CliRoute:
    """One route; composition defines matching and legacy argument adaptation."""
    name: str
    predicate: Callable[[list[str]], bool]
    load_handler: CommandFactory
    transform_args: Callable[[list[str]], list[str]] = _copy_args


def dispatch(routes: Iterable[CliRoute], argv, default: CommandFactory):
    """Use the first matching route, loading only its concrete handler.

    Ordered matching intentionally preserves existing CLI precedence. The
    router does not sort, remove flags or infer command names on its own.
    """
    arguments = list(argv)
    for route in routes:
        if route.predicate(list(arguments)):
            return route.load_handler()(route.transform_args(list(arguments)))
    return default()(arguments)


@dataclass(frozen=True)
class TaskSpec:
    """Technical metadata for a desktop task, supplied by application wiring.

    Titles, host attribute names and page assignments are opaque values here;
    the framework does not construct or import any business implementation.
    """
    key: str
    title: str
    hint: str
    page: int
    inspector: int
    tree_attr: str
    shortcut: str | None = None
    run_label: str = '运行计算'
    pages: tuple[int, ...] = ()

    def __post_init__(self):
        for name in ('key', 'title', 'hint', 'tree_attr', 'run_label'):
            if type(getattr(self, name)) is not str:
                raise ValueError('Task metadata must use strings: ' + name)
        if not self.key or not self.title or not self.tree_attr:
            raise ValueError('Task key, title and tree attribute must not be empty')
        if type(self.page) is not int or self.page < 0 or type(self.inspector) is not int or self.inspector < 0:
            raise ValueError('Task page and inspector indices must be nonnegative integers')
        if self.shortcut is not None and (type(self.shortcut) is not str or not self.shortcut):
            raise ValueError('Task shortcut must be a nonempty string or None')
        if type(self.pages) is not tuple or any(type(page) is not int or page < 0 for page in self.pages):
            raise ValueError('Task pages must be a tuple of nonnegative integers')
        if self.pages and (self.page not in self.pages or len(self.pages) != len(set(self.pages))):
            raise ValueError('Task pages must include the primary page exactly once')

    @property
    def page_indices(self):
        return self.pages or (self.page,)


class TaskRegistry:
    """Ordered, immutable task metadata with unambiguous page ownership."""
    def __init__(self, tasks: Iterable[TaskSpec]):
        self._tasks = tuple(tasks)
        by_key, by_page = {}, {}
        for task in self._tasks:
            if not isinstance(task, TaskSpec):
                raise TypeError('TaskRegistry entries must be TaskSpec instances')
            if task.key in by_key:
                raise ValueError('Duplicate task key: ' + task.key)
            by_key[task.key] = task
            for page in task.page_indices:
                if page in by_page:
                    raise ValueError('Duplicate task page: ' + str(page))
                by_page[page] = task
        self._by_key = MappingProxyType(by_key)
        self._by_page = MappingProxyType(by_page)

    def __iter__(self):
        return iter(self._tasks)

    def __len__(self):
        return len(self._tasks)

    def __getitem__(self, key):
        return self._by_key[key]

    def for_page(self, page, *, default=None):
        try:
            return self._by_page[page]
        except KeyError:
            if default is None:
                raise
            return self._by_key[default]
