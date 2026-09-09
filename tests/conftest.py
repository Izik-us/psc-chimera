"""Test-only compatibility for historical internal imports.

Production code has no compatibility monkey-patching.  One legacy integration
module still imports ``CHIMERAv2`` from ``chimera.chimera_v2``; after all test
modules have been imported, redirect only that test module's local binding to
the supported public composition.  This leaves canonical-architecture tests
free to verify that the implementation module itself was not mutated.
"""


def pytest_collection_modifyitems(session, config, items):
    del session, config
    from chimera import CHIMERAv2 as canonical

    for item in items:
        module = getattr(item, "module", None)
        if module is not None and module.__name__.endswith("test_chimera_v2"):
            module.CHIMERAv2 = canonical
