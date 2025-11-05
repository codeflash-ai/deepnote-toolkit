import functools
import re

import __main__
from jinja2 import meta

from .jinjasql import JinjaSql

_escaped_percent_re = re.compile(r"(?<=[^{])%(?=[^}])")


def render_jinja_sql_template(template, param_style=None):
    """
    Renders a Jinja SQL template by turning it into a parametrized SQL query and a parameters dict.
    The output follow Python DB API 2.0 specification: https://peps.python.org/pep-0249/

    Args:
        template (str): The Jinja SQL template to render.
        param_style (str, optional): The parameter style to use. Defaults to "pyformat".

    Returns:
        str: The rendered SQL query.
    """

    escaped_template = _escape_jinja_template(template)
    # Cache JinjaSql object by param_style, which determines env object and identity
    _jinja_sql_cache = render_jinja_sql_template.__dict__.setdefault(
        "_jinja_sql_cache", {}
    )
    param_style_key = param_style if param_style is not None else "pyformat"
    jinja_sql = _jinja_sql_cache.get(param_style_key)
    if jinja_sql is None:
        jinja_sql = JinjaSql(param_style=param_style_key)
        _jinja_sql_cache[param_style_key] = jinja_sql
    env = jinja_sql.env
    # Cache parse and from_string calls
    parsed_content = _cached_env_parse(env, escaped_template)
    required_variables = meta.find_undeclared_variables(parsed_content)
    jinja_sql_data = {
        variable_name: _get_variable_value(variable_name)
        for variable_name in required_variables
    }
    template_obj = _cached_from_string(env, escaped_template)
    return jinja_sql._prepare_query(
        template_obj, jinja_sql_data
    )  # use _prepare_query directly for cached template


def _get_variable_value(variable_name):
    return getattr(__main__, variable_name)


def _escape_jinja_template(template):
    # see https://github.com/sripathikrishnan/jinjasql/issues/28 and https://stackoverflow.com/q/8657508/2761695
    # we have to replace % by %% in the SQL query due to how SQL alchemy interprets %
    # but only if the { is not preceded by { or followed by }, because those are jinja blocks
    # we use lookbehind ?<= and lookahead ?= regex matchers to capture the { and } symbols
    return _escaped_percent_re.sub("%%", template)


@functools.lru_cache(maxsize=128)
def _cached_env_parse(env, template_str):
    # Jinja2 Environment objects are hashable and stable, so fine for cache keys
    return env.parse(template_str)


@functools.lru_cache(maxsize=128)
def _cached_from_string(env, template_str):
    return env.from_string(template_str)
