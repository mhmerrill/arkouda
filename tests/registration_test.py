import pytest
import json
from context import arkouda as ak
from base_test import ArkoudaTest
from arkouda.pdarrayclass import RegistrationError, unregister_pdarray_by_name
N = 100
UNIQUE = N // 4


class RegistrationTest(ArkoudaTest):
    
    def setUp(self):
        ArkoudaTest.setUp(self)
        self.a_array = ak.ones(10,dtype=ak.int64)
        self.b_array = ak.ones(10,dtype=ak.int64)

    def test_register(self):
        '''
        Tests the following:
        
        1. register invocation
        2. pdarray.name matches register name
        3. original and registered pdarray are equal
        4. method invocation on a cleared, registered array succeeds
        '''
        ar_array = self.a_array.register('test_int64_a')

        self.assertEqual('test_int64_a', self.a_array.name, "Expect name to change inplace")
        self.assertTrue(self.a_array is ar_array, "These should be the same object")
        self.assertTrue((self.a_array.to_ndarray() == ar_array.to_ndarray()).all())
        ak.clear()
        # Both ar_array and self.a_array point to the same object, so both should still be usable.
        str(ar_array)
        str(self.a_array)

        try:
            self.a_array.unregister()
        except (RuntimeError, RegistrationError):
            pass  # Will be tested in `test_unregister`

    def test_double_register(self):
        """
        Tests the case when two objects get registered using the same user_defined_name
        """
        a = ak.ones(3, dtype=ak.int64)
        b = ak.ones(3, dtype=ak.int64)
        b.fill(2)
        a.register("foo")

        with self.assertRaises(RegistrationError, msg="Should raise an Error"):
            b.register("foo")

        # Clean up the registry
        a.unregister()

    def test_registration_type_check(self):
        """
        Tests type checking of user_defined_name for register and attach
        """

        a = ak.ones(3, dtype=ak.int64)

        with self.assertRaises(TypeError, msg="register() should raise TypeError when user_defined_name is not a str"):
            a.register(7)
        with self.assertRaises(TypeError, msg="attach() should raise TypeError when user_defined_name is not a str"):
            a.attach(7)

        ak.clear()

    def test_unregister(self):
        '''
        Tests the following:
        
        1. unregister invocation
        2. method invocation on a cleared, unregistered array raises RuntimeError
        '''
        ar_array = self.a_array.register('test_int64_a') 
        
        self.assertEqual('[1 1 1 1 1 1 1 1 1 1]', str(ar_array))
        ar_array.unregister()
        self.assertEqual('[1 1 1 1 1 1 1 1 1 1]', str(ar_array))
        
        ak.clear()
        
        with self.assertRaises(RuntimeError):
            str(ar_array)
            
        with self.assertRaises(RuntimeError):            
            repr(ar_array)
    
    def test_attach(self):
        '''
        Tests the following:
        
        1. Attaching to a registered pdarray
        2. The registered and attached pdarrays are equal
        3. The attached pdarray is deleted server-side following
           unregister of registered pdarray and invocation of 
           ak.clear()
        4. method invocation on cleared attached array raises RuntimeError
        '''
        ar_array = self.a_array.register('test_int64_a')
        aar_array = ak.attach_pdarray('test_int64_a')
        
        self.assertEqual(ar_array.name, aar_array.name)
        self.assertTrue((ar_array.to_ndarray() == 
                                      aar_array.to_ndarray()).all())
        
        ak.disconnect()
        ak.connect(server=ArkoudaTest.server, port=ArkoudaTest.port)
        aar_array = ak.attach_pdarray('test_int64_a')
        
        self.assertEqual(ar_array.name, aar_array.name)
        self.assertTrue((ar_array.to_ndarray() == 
                                    aar_array.to_ndarray()).all())
        
        ar_array.unregister()
        ak.clear()
        
        with self.assertRaises(RuntimeError):            
            str(aar_array)
            
        with self.assertRaises(RuntimeError):            
            repr(aar_array)
    
    def test_clear(self): 
        '''
        Tests the following:
        
        1. clear() removes server-side pdarrays that are unregistered
        2. Registered pdarrays remain after ak.clear()
        3. All cleared pdarrays throw RuntimeError upon method invocation
        4. Method invocation on registered arrays succeeds after ak.clear()
        '''
        ar_array = self.a_array.register('test_int64_a')
        aar_array = self.a_array.register('test_int64_aa')

        self.assertTrue(ar_array is aar_array,
                        msg="With inplace modification, these should be the same")
        self.assertEqual(ar_array.name, "test_int64_aa",
                         msg="ar_array.name should be updated with inplace modification")

        twos_array = ak.ones(10,dtype=ak.int64).register('twos_array')
        twos_array.fill(2)
        
        g_twos_array = self.a_array + self.b_array
        self.assertTrue((twos_array.to_ndarray() == 
                                     g_twos_array.to_ndarray()).all())

        ak.clear() # This should remove self.b_array and g_twos_array

        with self.assertRaises(RuntimeError, msg="g_twos_array should have been cleared because it wasn't registered"):
            str(g_twos_array)

        with self.assertRaises(RuntimeError, msg="self.b_array should have been cleared because it wasn't registered"):
            str(self.b_array)

        # Assert these exist by invoking them and not receiving an exception
        str(self.a_array)
        str(ar_array)

        with self.assertRaises(RuntimeError, msg="Should raise error because self.b_array was cleared"):
            self.a_array + self.b_array

        g_twos_array = ar_array + aar_array
        self.assertTrue((twos_array.to_ndarray() == 
                                       g_twos_array.to_ndarray()).all())

    def test_register_info(self):
        '''
        Tests the following:

        1. json.loads(info(AllSymbols)) is an empty list when the symbol table is empty
        2. json.loads(info(RegisteredSymbols)) is an empty list when the registry is empty
        3. The registered field is set to false for objects that have not been registered
        4. The registered field is set to true for objects that have been registered
        5. info(ak.AllSymbols) contains both registered and non-registered objects
        6. info(ak.RegisteredSymbols) only contains registered objects
        7. info raises RunTimeError when called on objects not found in symbol table
        '''
        # Cleanup symbol table from previous tests
        cleanup()

        self.assertFalse(json.loads(ak.information(ak.AllSymbols)),
                         msg='info(AllSymbols) should be empty list')
        self.assertFalse(json.loads(ak.information(ak.RegisteredSymbols)),
                         msg='info(RegisteredSymbols) should be empty list')

        my_pdarray = ak.ones(10, dtype=ak.int64)
        self.assertFalse(json.loads(my_pdarray.info())[0]['registered'],
                         msg='my_array should be in all symbols but not be registered')

        # After register(), the registered field should be set to true for all info calls
        my_pdarray.register('keep_me')
        self.assertTrue(json.loads(ak.information('keep_me'))[0]['registered'],
                        msg='keep_me array not found or not registered')
        self.assertTrue(any([sym['registered'] for sym in json.loads(ak.information(ak.AllSymbols))]),
                        msg='No registered objects were found in symbol table')
        self.assertTrue(any([sym['registered'] for sym in json.loads(ak.information(ak.RegisteredSymbols))]),
                        msg='No registered objects were found in registry')

        not_registered_array = ak.ones(10, dtype=ak.int64)
        self.assertTrue(len(json.loads(ak.information(ak.AllSymbols))) > len(json.loads(ak.information(ak.RegisteredSymbols))),
                        msg='info(AllSymbols) should have more objects than info(RegisteredSymbols) before clear()')
        ak.clear()
        self.assertTrue(len(json.loads(ak.information(ak.AllSymbols))) == len(json.loads(ak.information(ak.RegisteredSymbols))),
                        msg='info(AllSymbols) and info(RegisteredSymbols) should have same num of objects after clear()')

        # After unregister(), the registered field should be set to false for AllSymbol and object name info calls
        # RegisteredSymbols info calls should return ak.EmptyRegistry
        my_pdarray.unregister()
        self.assertFalse(any([obj['registered'] for obj in json.loads(my_pdarray.info())]),
                        msg='info(my_array) registered field should be false after unregister()')
        self.assertFalse(all([obj['registered'] for obj in json.loads(ak.information(ak.AllSymbols))]),
                        msg='info(AllSymbols) should contain unregistered objects')
        self.assertFalse(json.loads(ak.information(ak.RegisteredSymbols)),
                         msg='info(RegisteredSymbols) empty list failed after unregister()')

        ak.clear()
        # RuntimeError when calling info on an object not in the symbol table
        with self.assertRaises(RuntimeError, msg="RuntimeError for info on object not in symbol table"):
            ak.information('keep_me')
        self.assertFalse(json.loads(ak.information(ak.AllSymbols)),
                         msg='info(AllSymbols) should be empty list')
        self.assertFalse(json.loads(ak.information(ak.RegisteredSymbols)),
                         msg='info(RegisteredSymbols) should be empty list')
        cleanup()

    def test_in_place_info(self):
        """
        Tests the class level info method for pdarray, String, and Categorical
        """
        cleanup()
        my_pda = ak.ones(10, ak.int64)
        self.assertFalse(any([sym['registered'] for sym in json.loads(my_pda.info())]),
                        msg='no components of my_pda should be registered before register call')
        my_pda.register('my_pda')
        self.assertTrue(all([sym['registered'] for sym in json.loads(my_pda.info())]),
                        msg='all components of my_pda should be registered after register call')

        my_str = ak.random_strings_uniform(1, 10, UNIQUE, characters='printable')
        self.assertFalse(any([sym['registered'] for sym in json.loads(my_str.info())]),
                        msg='no components of my_str should be registered before register call')
        my_str.register('my_str')
        self.assertTrue(all([sym['registered'] for sym in json.loads(my_str.info())]),
                        msg='all components of my_str should be registered after register call')

        my_cat = ak.Categorical(ak.array([f"my_cat {i}" for i in range(1, 11)]))
        self.assertFalse(any([sym['registered'] for sym in json.loads(my_cat.info())]),
                        msg='no components of my_cat should be registered before register call')
        my_cat.register('my_cat')
        self.assertTrue(all([sym['registered'] for sym in json.loads(my_cat.info())]),
                        msg='all components of my_cat should be registered after register call')
        cleanup()


    def test_is_registered(self):
        """
        Tests the pdarray.is_registered() function
        """
        cleanup()
        a = ak.ones(10, dtype=ak.int64)
        self.assertFalse(a.is_registered())

        a.register('keep')
        self.assertTrue(a.is_registered())

        a.unregister()
        self.assertFalse(a.is_registered())
        ak.clear()

    def test_list_registry(self):
        """
        Tests the generic ak.list_registry() function
        """
        cleanup()
        # Test list_registry when the symbol table is empty
        self.assertFalse(ak.list_registry(), "registry should be empty")

        a = ak.ones(10, dtype=ak.int64)
        # list_registry() should return an empty list which is implicitly False
        self.assertFalse(ak.list_registry())

        a.register('keep')
        self.assertTrue('keep' in ak.list_registry())
        cleanup()

    def test_string_registration_suite(self):
        cleanup()
        # Initial registration should set name
        keep = ak.random_strings_uniform(1, 10, UNIQUE, characters='printable')
        self.assertTrue(keep.register("keep_me").name == "keep_me")
        self.assertTrue(keep.is_registered(), "Expected Strings object to be registered")

        # Register a second time to confirm name change
        self.assertTrue(keep.register("kept").name == "kept")
        self.assertTrue(keep.is_registered(), "Object should be registered with updated name")

        # Add an item to discard, confirm our registered item remains and discarded item is gone
        discard = ak.random_strings_uniform(1, 10, UNIQUE, characters='printable')
        ak.clear()
        self.assertTrue(keep.name == "kept")
        with self.assertRaises(RuntimeError, msg="discard was not registered and should be discarded"):
            str(discard)

        # Unregister, should remain usable until we clear
        keep.unregister()
        str(keep) # Should not cause error
        self.assertFalse(keep.is_registered(), "This item should no longer be registered")
        ak.clear()
        with self.assertRaises(RuntimeError, msg="keep was unregistered and should be cleared"):
            str(keep) # should cause RuntimeError

        # Test attach functionality
        s1 = ak.random_strings_uniform(1, 10, UNIQUE, characters='printable')
        self.assertTrue(s1.register("uut").is_registered(), "uut should be registered")
        s1 = None
        self.assertTrue(s1 is None, "Reference should be cleared")
        s1 = ak.Strings.attach("uut")
        self.assertTrue(s1.is_registered(), "Should have re-attached to registered object")
        str(s1)  # This will throw an exception if the object doesn't exist server-side

        # Test the Strings unregister by name using previously registered object
        ak.Strings.unregister_strings_by_name("uut")
        self.assertFalse(s1.is_registered(), "Expected object to be unregistered")
        cleanup()

    def test_string_is_registered(self):
        """
        Tests the Strings.is_registered() function
        """
        keep = ak.random_strings_uniform(1, 10, UNIQUE, characters='printable')
        self.assertFalse(keep.is_registered())

        keep.register('keep_me')
        self.assertTrue(keep.is_registered())

        keep.unregister()
        self.assertFalse(keep.is_registered())

        ak.clear()

    def test_delete_registered(self):
        """
        Tests the following:

        1. delete cmd doesn't delete registered objects and returns appropriate message
        2. delete cmd does delete non-registered objects and returns appropriate message
        3. delete cmd raises RuntimeError for unknown symbols
        """
        cleanup()
        a = ak.ones(3, dtype=ak.int64)
        b = ak.ones(3, dtype=ak.int64)

        # registered objects are not deleted from symbol table
        a.register('keep')
        self.assertEqual(ak.client.generic_msg(cmd='delete', args=a.name),
                         f'registered symbol, {a.name}, not deleted')
        self.assertTrue(a.name in ak.list_symbol_table())

        # non-registered objects are deleted from symbol table
        self.assertEqual(ak.client.generic_msg(cmd='delete', args=b.name),
                         'deleted ' + b.name)
        self.assertTrue(b.name not in ak.list_symbol_table())

        # RuntimeError when calling delete on an object not in the symbol table
        with self.assertRaises(RuntimeError):
            ak.client.generic_msg(cmd='delete', args='not_in_table')

    def test_categorical_registration_suite(self):
        """
        Test register, is_registered, attach, unregister, unregister_categorical_by_name
        """
        cleanup()  # Make sure we start with a clean registry
        c = ak.Categorical(ak.array([f"my_cat {i}" for i in range(1, 11)]))
        self.assertFalse(c.is_registered(), "test_me should be unregistered")
        self.assertTrue(c.register("test_me").is_registered(), "test_me categorical should be registered")
        c = None  # Should trigger destructor, but survive server deletion because it is registered
        self.assertTrue(c is None, "The reference to `c` should be None")
        c = ak.Categorical.attach("test_me")
        self.assertTrue(c.is_registered(), "test_me categorical should be registered after attach")
        c.unregister()
        self.assertFalse(c.is_registered(), "test_me should be unregistered")
        self.assertTrue(c.register("another_name").name == "another_name" and c.is_registered())

        # Test static unregister_by_name
        ak.Categorical.unregister_categorical_by_name("another_name")
        self.assertFalse(c.is_registered(), "another_name should be unregistered")

        # now mess with the subcomponents directly to test is_registered mis-match logic
        c.register("another_name")
        unregister_pdarray_by_name("another_name.codes")
        with pytest.raises(RegistrationError):
            c.is_registered()

    def test_categorical_from_codes_registration_suite(self):
        """
        Test register, is_registered, attach, unregister, unregister_categorical_by_name
        for Categorical made using .from_codes
        """
        cleanup()  # Make sure we start with a clean registry
        categories = ak.array(['a', 'b', 'c'])
        codes = ak.array([0, 1, 0, 2, 1])
        cat = ak.Categorical.from_codes(codes, categories)
        self.assertFalse(cat.is_registered(), "test_me should be unregistered")
        self.assertTrue(cat.register("test_me").is_registered(), "test_me categorical should be registered")
        cat = None  # Should trigger destructor, but survive server deletion because it is registered
        self.assertTrue(cat is None, "The reference to `c` should be None")
        cat = ak.Categorical.attach("test_me")
        self.assertTrue(cat.is_registered(), "test_me categorical should be registered after attach")
        cat.unregister()
        self.assertFalse(cat.is_registered(), "test_me should be unregistered")
        self.assertTrue(cat.register("another_name").name == "another_name" and cat.is_registered())

        # Test static unregister_by_name
        ak.Categorical.unregister_categorical_by_name("another_name")
        self.assertFalse(cat.is_registered(), "another_name should be unregistered")

        # now mess with the subcomponents directly to test is_registered mis-match logic
        cat.register("another_name")
        unregister_pdarray_by_name("another_name.codes")
        with pytest.raises(RegistrationError):
            cat.is_registered()

    def test_attach_weak_binding(self):
        """
        Ultimately pdarrayclass issues delete calls to the server when a bound object goes out of scope, if you bind
        to a server object more than once and one of those goes out of scope it affects all other references to it.
        """
        cleanup()
        a = ak.ones(3, dtype=ak.int64).register("a_reg")
        self.assertTrue(str(a), "Expected to pass")
        b = ak.attach_pdarray("a_reg")
        b.unregister()
        b = None  # Force out of scope
        with self.assertRaises(RuntimeError):
            str(a)

def cleanup():
    ak.clear()
    for registered_name in ak.list_registry():
        ak.unregister_pdarray_by_name(registered_name)
    ak.clear()
