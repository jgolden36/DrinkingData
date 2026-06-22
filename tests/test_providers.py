"""Provider normalization: the treated-set attribution rules (§2.1)."""

import pytest

from drinkingdata.data import providers


@pytest.mark.parametrize(
    "org, expected",
    [
        ("Anthropic", "Google"),        # lab → hosting hyperscaler (§2.1)
        ("Google DeepMind", "Google"),
        ("OpenAI", "Microsoft"),
        ("Microsoft Azure", "Microsoft"),
        ("Amazon Web Services", "Amazon"),
        ("Meta AI", "Meta"),
        ("Facebook", "Meta"),
    ],
)
def test_primary_attribution(org, expected):
    primary, _ = providers.map_to_hyperscaler(org)
    assert primary == expected


def test_unknown_and_missing_return_none():
    assert providers.map_to_hyperscaler(None) == (None, None)
    assert providers.map_to_hyperscaler("") == (None, None)
    assert providers.map_to_hyperscaler("Equinix Colocation") == (None, None)


def test_multi_provider_string_gives_secondary():
    primary, secondary = providers.map_to_hyperscaler("OpenAI on Amazon AWS")
    assert {primary, secondary} == {"Microsoft", "Amazon"}


def test_is_hyperscaler_flag():
    assert providers.is_hyperscaler("Google")
    assert not providers.is_hyperscaler("Some Regional Telco")


def test_hyperscaler_set_is_the_four():
    assert set(providers.HYPERSCALERS) == {"Google", "Microsoft", "Amazon", "Meta"}
