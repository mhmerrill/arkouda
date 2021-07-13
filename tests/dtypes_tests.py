import numpy as np
from context import arkouda as ak
from arkouda import dtypes
from base_test import ArkoudaTest

'''
DtypesTest encapsulates arkouda dtypes module methods
'''
class DtypesTest(ArkoudaTest):

    def test_check_np_dtype(self):
      
        '''
        Tests dtypes.check_np_dtype method 
        
        :return: None
        :raise: AssertionError if 1.. test cases fail
        '''
        dtypes.check_np_dtype(np.dtype(np.bool))
        dtypes.check_np_dtype(np.dtype(np.int64))
        dtypes.check_np_dtype(np.dtype(np.float64))
        dtypes.check_np_dtype(np.dtype(np.uint8))
        dtypes.check_np_dtype(np.dtype(np.str))

        with self.assertRaises(TypeError):
            dtypes.check_np_dtype(np.dtype(np.int16))
        with self.assertRaises(TypeError):
            dtypes.check_np_dtype('np.str')

    def test_translate_np_dtype(self):
        '''
        Tests dtypes.translate_np_dtype method
        
        :return: None
        :raise: AssertionError if 1.. test cases fail
        '''
        d_tuple = dtypes.translate_np_dtype(np.dtype(np.bool))
        self.assertEqual(1, d_tuple[1])
        self.assertEqual('bool', d_tuple[0])
        d_tuple = dtypes.translate_np_dtype(np.dtype(np.int64))
        self.assertEqual(8, d_tuple[1])
        self.assertEqual('int', d_tuple[0])  
        d_tuple = dtypes.translate_np_dtype(np.dtype(np.float64))
        self.assertEqual(8, d_tuple[1])
        self.assertEqual('float', d_tuple[0])        
        d_tuple = dtypes.translate_np_dtype(np.dtype(np.uint8))
        self.assertEqual(1, d_tuple[1])
        self.assertEqual('uint',d_tuple[0])    
        d_tuple = dtypes.translate_np_dtype(np.dtype(np.str))
        self.assertEqual(0, d_tuple[1])
        self.assertEqual('str', d_tuple[0])   

        with self.assertRaises(TypeError):
            dtypes.check_np_dtype(np.dtype(np.int16))
        with self.assertRaises(TypeError):
            dtypes.translate_np_dtype('np.str')
            
    def test_resolve_scalar_dtype(self):
        '''
        Tests dtypes.resolve_scalar_dtype method
        
        :return: None
        :raise: AssertionError if 1.. test cases fail
        '''
        self.assertEqual('bool', dtypes.resolve_scalar_dtype(True))
        self.assertEqual('int64', dtypes.resolve_scalar_dtype(1))
        self.assertEqual('float64', dtypes.resolve_scalar_dtype(float(0.0)))
        self.assertEqual('str', dtypes.resolve_scalar_dtype('test'))
        self.assertEqual('int64', dtypes.resolve_scalar_dtype(np.int64(1))) 
        self.assertEqual("<class 'list'>", dtypes.resolve_scalar_dtype([1]))
        
    def test_pdarrays_datatypes(self):
        self.assertEqual(dtypes.dtype('float64'), ak.ones(10).dtype)
        self.assertEqual(dtypes.dtype('str'), 
                         ak.array(['string {}'.format(i) for i in range(0,10)]).dtype)

    def test_isSupportedInt(self):
        '''
        Tests for both True and False scenarios of the isSupportedInt method.
        '''
        self.assertTrue(dtypes.isSupportedInt(1))
        self.assertTrue(dtypes.isSupportedInt(np.int64(1)))
        self.assertTrue(dtypes.isSupportedInt(np.int64(1.0)))
        self.assertFalse(dtypes.isSupportedInt(1.0))
        self.assertFalse(dtypes.isSupportedInt('1'))
        self.assertFalse(dtypes.isSupportedInt('1.0'))
        
    def test_isSupportedFloat(self):
        '''
        Tests for both True and False scenarios of the isSupportedFloat method.
        '''
        self.assertTrue(dtypes.isSupportedFloat(1.0))
        self.assertTrue(dtypes.isSupportedFloat(float(1)))
        self.assertTrue(dtypes.isSupportedFloat(np.float64(1.0)))
        self.assertTrue(dtypes.isSupportedFloat(np.float64(1)))
        self.assertFalse(dtypes.isSupportedFloat(np.int64(1.0)))
        self.assertFalse(dtypes.isSupportedFloat(int(1.0)))
        self.assertFalse(dtypes.isSupportedFloat('1'))
        self.assertFalse(dtypes.isSupportedFloat('1.0'))
        
    def test_DtypeEnum(self):
        '''
        Tests for DTypeEnum, ak.DTypes, and ak.ARKOUDA_SUPPORTED_DTYPES
        '''
        self.assertEqual('bool', str(dtypes.DType.BOOL))
        self.assertEqual('float', str(dtypes.DType.FLOAT))
        self.assertEqual('float64', str(dtypes.DType.FLOAT64))
        self.assertEqual('int', str(dtypes.DType.INT))
        self.assertEqual('int64', str(dtypes.DType.INT64))
        self.assertEqual('str', str(dtypes.DType.STR))
        self.assertEqual('uint8', str(dtypes.DType.UINT8))
        self.assertEqual(frozenset({'float','float64', 'bool', 'uint8', 
                                    'int','int64', 'str'}), ak.DTypes)
        self.assertEqual(frozenset({'float','float64', 'bool', 'uint8', 
                          'int','int64', 'str'}), ak.ARKOUDA_SUPPORTED_DTYPES)
        
    def test_NumericDTypes(self):
        self.assertEqual(frozenset(['bool', 'float', 'float64','int','int64']), 
                         dtypes.NumericDTypes)
        
    def test_SeriesDTypes(self):
        self.assertEqual(np.str_, dtypes.SeriesDTypes['string'])
        self.assertEqual(np.str_, dtypes. SeriesDTypes["<class 'str'>"])
        self.assertEqual(np.int64, dtypes. SeriesDTypes['int64'])
        self.assertEqual(np.int64, dtypes. SeriesDTypes["<class 'numpy.int64'>"])
        self.assertEqual(np.float64, dtypes. SeriesDTypes['float64'])
        self.assertEqual(np.float64, dtypes. SeriesDTypes["<class 'numpy.float64'>"])
        self.assertEqual(np.bool, dtypes. SeriesDTypes['bool'])
        self.assertEqual(np.bool, dtypes. SeriesDTypes["<class 'bool'>"])
        self.assertEqual(np.int64, dtypes. SeriesDTypes['datetime64[ns]'])
        self.assertEqual(np.int64, dtypes. SeriesDTypes['timedelta64[ns]'])

    def test_scalars(self):
        self.assertEqual("<class 'bool'>", str(ak.bool_scalars))
        self.assertEqual('typing.Union[float, numpy.float64]', str(ak.float_scalars))
        self.assertEqual('typing.Union[int, numpy.int64]', str(ak.int_scalars))
        self.assertEqual('typing.Union[float, numpy.float64, int, numpy.int64]', 
                         str(ak.numeric_scalars))
        self.assertEqual('typing.Union[str, numpy.str_]', str(ak.str_scalars))
        self.assertEqual('typing.Union[numpy.float64, numpy.int64, bool, numpy.uint8, str, numpy.str_]', 
                         str(ak.numpy_scalars))
        self.assertEqual('typing.Union[float, numpy.float64, int, numpy.int64, bool, str, numpy.str_]', 
                         str(ak.all_scalars))
        
    def test_number_format_strings(self):
        self.assertEqual('{}', dtypes.NUMBER_FORMAT_STRINGS['bool'])
        self.assertEqual('{:n}', dtypes.NUMBER_FORMAT_STRINGS['int64'])
        self.assertEqual('{:.17f}', dtypes.NUMBER_FORMAT_STRINGS['float64'])
        self.assertEqual('f', dtypes.NUMBER_FORMAT_STRINGS['np.float64'])
        self.assertEqual('{:n}', dtypes.NUMBER_FORMAT_STRINGS['uint8'])
