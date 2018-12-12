from unittest import TestCase

from snips_nlu.common.registrable import Registrable
from snips_nlu.exceptions import NotRegisteredError, AlreadyRegisteredError


class TestRegistrable(TestCase):
    def test_should_register_subclass(self):
        # Given
        class MyBaseClass(Registrable):
            pass

        @MyBaseClass.register("first_subclass")
        class MyFirstSubclass(MyBaseClass):
            pass

        @MyBaseClass.register("second_subclass")
        class MySecondSubclass(MyBaseClass):
            pass

        # When
        my_subclass = MyBaseClass.by_name("second_subclass")

        # Then
        self.assertEqual(MySecondSubclass, my_subclass)

    def test_should_raise_when_not_registered(self):
        # Given
        class MyBaseClass(Registrable):
            pass

        # When / Then
        with self.assertRaises(NotRegisteredError):
            MyBaseClass.by_name("my_unregistered_subclass")

    def test_should_raise_when_already_registered(self):
        # Given
        class MyBaseClass(Registrable):
            pass

        @MyBaseClass.register("my_duplicated_subclass")
        class MySubclass(MyBaseClass):
            pass

        # When / Then
        with self.assertRaises(AlreadyRegisteredError):
            @MyBaseClass.register("my_duplicated_subclass")
            class MySecondSubclass(MyBaseClass):
                pass

    def test_should_override_already_registered_subclass(self):
        # Given
        class MyBaseClass(Registrable):
            pass

        @MyBaseClass.register("my_subclass")
        class MyOverridenSubclass(MyBaseClass):
            pass

        @MyBaseClass.register("my_subclass", override=True)
        class MySubclass(MyBaseClass):
            pass

        # When
        subclass = MyBaseClass.by_name("my_subclass")

        # Then
        self.assertEqual(MySubclass, subclass)
