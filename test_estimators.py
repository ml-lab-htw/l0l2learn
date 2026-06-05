import functools

from sklearn.utils.estimator_checks import estimator_checks_generator
from l0l2learn import L0L2Classifier, ResampledL0L2Classifier


def check_name(check):
    if isinstance(check, functools.partial):
        return check.func.__name__
    return getattr(check, "__name__", repr(check))


def list_failures(estimator):
    print(f"\n{estimator.__class__.__name__}")
    print("=" * 80)

    n_failed = 0

    for estimator_instance, check in estimator_checks_generator(estimator):
        try:
            check(estimator_instance)
        except Exception as e:
            n_failed += 1
            print(f"FAILED: {check_name(check)}")
            print(type(e).__name__)
            print(str(e) or "<no message>")
            print("-" * 80)

    print(f"Total failures: {n_failed}")


def main():
    estimators = [
        L0L2Classifier(),
        ResampledL0L2Classifier(b=3),
    ]

    for estimator in estimators:
        list_failures(estimator)


if __name__ == "__main__":
    main()
