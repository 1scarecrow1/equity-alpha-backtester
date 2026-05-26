from typing import Any


class DataExpression:
    def __init__(self, expression: str, variable_names: set[str]):
        self.expression = expression
        self.variable_names = set(variable_names)

    def __repr__(self) -> str:
        return self.expression

    def __str__(self) -> str:
        return self.expression

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        def expression_method(*args: Any, **kwargs: Any) -> "DataExpression":
            rendered_args = [as_expression(arg) for arg in args]
            rendered_kwargs = {
                key: as_expression(value)
                for key, value in kwargs.items()
            }
            variables = set(self.variable_names)
            for expression in [*rendered_args, *rendered_kwargs.values()]:
                variables.update(expression.variable_names)

            rendered = [expression.expression for expression in rendered_args]
            rendered.extend(
                f"{key}={expression.expression}"
                for key, expression in rendered_kwargs.items()
            )
            return DataExpression(f"{self.expression}.{name}({', '.join(rendered)})", variables)

        return expression_method
    
    def __call__(self, *args: Any, **kwargs: Any):
        raise TypeError("Evaluate DataExpression through strategy.build_alpha.evaluate_expression")

    def _binary(self, other: Any, operator: str) -> "DataExpression":
        other_expr = as_expression(other)
        return DataExpression(
            f"({self.expression} {operator} {other_expr.expression})",
            self.variable_names | other_expr.variable_names,
        )

    def _rbinary(self, other: Any, operator: str) -> "DataExpression":
        other_expr = as_expression(other)
        return DataExpression(
            f"({other_expr.expression} {operator} {self.expression})",
            self.variable_names | other_expr.variable_names,
        )

    def __add__(self, other: Any) -> "DataExpression":
        return self._binary(other, "+")

    def __radd__(self, other: Any) -> "DataExpression":
        return self._rbinary(other, "+")

    def __sub__(self, other: Any) -> "DataExpression":
        return self._binary(other, "-")

    def __rsub__(self, other: Any) -> "DataExpression":
        return self._rbinary(other, "-")

    def __mul__(self, other: Any) -> "DataExpression":
        return self._binary(other, "*")

    def __rmul__(self, other: Any) -> "DataExpression":
        return self._rbinary(other, "*")

    def __truediv__(self, other: Any) -> "DataExpression":
        return self._binary(other, "/")

    def __rtruediv__(self, other: Any) -> "DataExpression":
        return self._rbinary(other, "/")

    def __pow__(self, other: Any) -> "DataExpression":
        return self._binary(other, "**")

    def __rpow__(self, other: Any) -> "DataExpression":
        return self._rbinary(other, "**")

    def __mod__(self, other: Any) -> "DataExpression":
        return self._binary(other, "%")

    def __rmod__(self, other: Any) -> "DataExpression":
        return self._rbinary(other, "%")

    def __neg__(self) -> "DataExpression":
        return DataExpression(f"(-{self.expression})", set(self.variable_names))

    def __pos__(self) -> "DataExpression":
        return DataExpression(f"(+{self.expression})", set(self.variable_names))
    
    def _compare(self, other: Any, operator: str) -> "DataExpression":
        other_expr = as_expression(other)
        return DataExpression(
            f"({self.expression} {operator} {other_expr.expression})",
            self.variable_names | other_expr.variable_names,
        )
    
    def __gt__(self, other: Any) -> "DataExpression":
        return self._compare(other, ">")
    
    def __ge__(self, other: Any) -> "DataExpression":
        return self._compare(other, ">=")
    
    def __lt__(self, other: Any) -> "DataExpression":
        return self._compare(other, "<")
    
    def __le__(self, other: Any) -> "DataExpression":
        return self._compare(other, "<=")
    
    def __and__(self, other: Any) -> "DataExpression":
        return self._binary(other, "&")
    
    def __or__(self, other: Any) -> "DataExpression":
        return self._binary(other, "|")
    
def as_expression(value: Any) -> DataExpression:
    if isinstance(value, DataExpression):
        return value
    if isinstance(value, int | float | bool):
        return DataExpression(repr(value), set())
    raise TypeError(f"Cannot use {type(value).__name__} in a data expression")

