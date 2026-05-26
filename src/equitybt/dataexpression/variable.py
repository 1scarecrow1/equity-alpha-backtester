from equitybt.data_fields import get_variable_source
from equitybt.dataexpression import Any, DataExpression


class Variable(DataExpression):
    def __init__(self, name: str):
        super().__init__(name, {name})
        self.name = name

    @property
    def source(self) -> dict[str, Any]:
        return get_variable_source(self.name)

    @property
    def column_name(self) -> str:
        return self.source["column_name"]

    @property
    def description(self) -> str:
        return self.source["description"]

    @property
    def derived(self) -> dict[str, Any] | None:
        return self.source.get("derived")

    @property
    def is_custom(self) -> bool:
        return bool(self.source.get("is_custom"))
