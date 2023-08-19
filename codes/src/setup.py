from setuptools import setup, Extension
import numpy

setup(name='fixmesh',
      version='3.0',
      ext_modules =[Extension('_fixmesh',
                             ['fixmesh.c', 'fixmesh.i'],
                    include_dirs = [numpy.get_include(),'.'])
                   ])
