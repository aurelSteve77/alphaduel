"""Tests for the thread-safe Singleton metaclass."""

from __future__ import annotations

from alphaduel.utils.singleton import Singleton


def test_same_instance_returned():
    class Foo(metaclass=Singleton):
        def __init__(self, value=0):
            self.value = value

    a = Foo(1)
    b = Foo(2)
    assert a is b
    # Constructor args are ignored after first construction.
    assert a.value == 1


def test_distinct_classes_have_distinct_instances():
    class A(metaclass=Singleton):
        pass

    class B(metaclass=Singleton):
        pass

    assert A() is A()
    assert B() is B()
    assert A() is not B()
