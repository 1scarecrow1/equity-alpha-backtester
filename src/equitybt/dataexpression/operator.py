from equitybt.dataexpression.dataexpression import DataExpression, as_expression
from functools import update_wrapper
from typing import Any


class Operator:
    def __init__(self, function, name: str | None = None):
        self.function = function
        self.name = name or function.__name__
        update_wrapper(self, function)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not any(isinstance(value, DataExpression) for value in [*args, *kwargs.values()]):
            return self.function(*args, **kwargs)

        rendered_args = [_render_argument(arg) for arg in args]
        rendered_kwargs = {
            name: _render_argument(value)
            for name, value in kwargs.items()
        }
        variables = set()
        for _, expression_variables in [*rendered_args, *rendered_kwargs.values()]:
            variables.update(expression_variables)

        rendered = [expression for expression, _ in rendered_args]
        rendered.extend(
            f"{name}={expression}"
            for name, (expression, _) in rendered_kwargs.items()
        )
        return DataExpression(f"{self.name}({', '.join(rendered)})", variables)


def _render_argument(value: Any) -> tuple[str, set[str]]:
    if isinstance(value, DataExpression):
        return value.expression, set(value.variable_names)

    try:
        expression = as_expression(value)
    except TypeError:
        return repr(value), set()

    return expression.expression, set(expression.variable_names)
