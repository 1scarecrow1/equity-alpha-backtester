import re
from equitybt.dataexpression.dataexpression import DataExpression
from equitybt.dataexpression.variable import Variable

ExpressionInput = DataExpression | list[DataExpression] | tuple[DataExpression, ...] | dict[str, DataExpression]

TRADE_WHEN = "trade_when"

class Alpha:
    signal = Variable("signal")

    def __init__(
        self,
        signal: ExpressionInput | None = None,
        signals: ExpressionInput | None = None,
        constraint: ExpressionInput | None = None,
        constraints: ExpressionInput | None = None,
        exit_condition: DataExpression | int | None = -1,
    ):
        self._signals = _normalize_expression_inputs(
            singular=signal,
            plural=signals,
            singular_label="signal",
            plural_label="signals",
        )
        self._constraints = _normalize_expression_inputs(
            singular=constraint,
            plural=constraints,
            singular_label="constraint",
            plural_label="constraints",
        )
        self.exit_condition = exit_condition
        self._absorb_trade_when_signals()

    @property
    def signals(self) -> dict[str, DataExpression]:
        return dict(self._signals)

    @signals.setter
    def signals(self, signals: ExpressionInput | None):
        self._signals = _as_expression_map(signals, default_name="signal")

    @property
    def constraints(self) -> dict[str, DataExpression]:
        return dict(self._constraints)

    @constraints.setter
    def constraints(self, constraints: ExpressionInput | None):
        self._constraints = _as_expression_map(constraints, default_name="constraint")

    @property
    def exit_condition(self) -> DataExpression:
        return self._exit_condition or DataExpression("-1", set())

    @exit_condition.setter
    def exit_condition(self, exit_condition: DataExpression | int | None):
        if exit_condition is None or exit_condition == -1:
            self._exit_condition = None
            return
        if not isinstance(exit_condition, DataExpression):
            raise TypeError("exit_condition must be -1 or a DataExpression")
        self._exit_condition = exit_condition

    @property
    def fields(self) -> list[Variable]:
        variable_names: set[str] = set()
        expressions = [*self.signals.values(), *self.constraints.values()]
        if self.has_exit_condition:
            expressions.append(self._exit_condition)
        for expression in expressions:
            variable_names.update(expression.variable_names)

        return [
            Variable(name)
            for name in sorted(variable_names - self.generated_field_names)
        ]

    @property
    def has_exit_condition(self) -> bool:
        return self._exit_condition is not None

    @property
    def generated_field_names(self) -> set[str]:
        return set(self.signals)

    def _absorb_trade_when_signals(self) -> None:
        for name, signal in list(self._signals.items()):
            decomposed = _decompose_trade_when(signal)
            if decomposed is None:
                continue

            constraint, signal, exit_condition = decomposed
            self._signals[name] = signal

            if constraint is not None:
                base = "constraint" if name == "signal" else f"{name}_constraint"
                key, suffix = base, 1
                while key in self._constraints:
                    key, suffix = f"{base}_{suffix}", suffix + 1
                self._constraints[key] = constraint

            if exit_condition is not None and self._exit_condition is None:
                self.exit_condition = exit_condition

    def add_signal(self, name: str, signal: DataExpression):
        self._signals[name] = _validate_expression(signal, "signal")
        return self

    def add_constraint(self, name: str, constraint: DataExpression):
        self._constraints[name] = _validate_expression(constraint, "constraint")
        return self

    def update_signals(self, signals: ExpressionInput | None = None, **named_signals):
        self._signals.update(_as_expression_map(signals, default_name="signal"))
        self._signals.update(_validate_expression_map(named_signals, "signal"))
        return self

    def update_constraints(self, constraints: ExpressionInput | None = None, **named_constraints):
        self._constraints.update(_as_expression_map(constraints, default_name="constraint"))
        self._constraints.update(_validate_expression_map(named_constraints, "constraint"))
        return self

    def replace_signals(self, signals: ExpressionInput | None = None):
        self.signals = signals
        return self

    def replace_constraints(self, constraints: ExpressionInput | None = None):
        self.constraints = constraints
        return self

def _as_expression_map(
    expressions: ExpressionInput | None,
    *,
    default_name: str,
) -> dict[str, DataExpression]:
    if expressions is None:
        return {}

    if isinstance(expressions, DataExpression):
        return {default_name: expressions}

    if isinstance(expressions, dict):
        return _validate_expression_map(expressions, default_name)

    if isinstance(expressions, (list, tuple)):
        return {
            f"{default_name}_{index}": expression
            for index, expression in enumerate(expressions)
        }

    raise TypeError(f"Unsupported {default_name} input: {type(expressions).__name__}")


def _normalize_expression_inputs(
    *,
    singular: ExpressionInput | None,
    plural: ExpressionInput | None,
    singular_label: str,
    plural_label: str,
) -> dict[str, DataExpression]:
    if singular is not None and plural is not None:
        raise ValueError(f"Pass either {singular_label} or {plural_label}, not both.")

    expressions = plural if plural is not None else singular
    return _as_expression_map(expressions, default_name=singular_label)


def _validate_expression(expression, default_name: str) -> DataExpression:
    if not isinstance(expression, DataExpression):
        raise TypeError(f"Expected {default_name} to be a DataExpression, got {type(expression).__name__}")
    return expression


def _validate_expression_map(expressions: dict[str, DataExpression], default_name: str) -> dict[str, DataExpression]:
    return {
        name: _validate_expression(expression, default_name)
        for name, expression in expressions.items()
    }


def _decompose_trade_when(
    signal: DataExpression,
) -> tuple[DataExpression | None, DataExpression, DataExpression | None] | None:
    
    if not isinstance(signal, DataExpression):
        return None

    inner = _matching_call_inner(signal.expression, TRADE_WHEN)
    if inner is None:
        return None

    args = _split_top_level_args(inner)
    if len(args) < 2:
        return None

    constraint = _sub_expression(args[0], signal) if args[0] not in ("-1", "True") else None
    signal = _sub_expression(args[1], signal)
    exit_condition = (
        _sub_expression(args[2], signal)
        if len(args) >= 3 and args[2] not in ("-1", "")
        else None
    )
    return constraint, signal, exit_condition


def _matching_call_inner(text: str, name: str) -> str | None:
    text = text.strip()
    prefix = f"{name}("
    if not text.startswith(prefix) or not text.endswith(")"):
        return None

    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[len(prefix):index] if index == len(text) - 1 else None
    return None


def _split_top_level_args(inner: str) -> list[str]:
    args: list[str] = []
    current: list[str] = []
    depth = 0
    for char in inner:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        args.append("".join(current).strip())
    return args


def _sub_expression(text: str, parent: DataExpression) -> DataExpression:
    tokens = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text))
    variables = {name for name in parent.variable_names if name in tokens}
    return DataExpression(text, variables)
