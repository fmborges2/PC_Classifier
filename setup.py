from setuptools import setup

with open("README.md", "r") as arq:
    readme = arq.read()

setup(name='ocpc_py',
<<<<<<< HEAD
    version='0.1.4',
=======
    version='0.1.3',
>>>>>>> a3593666f52218c9861eff9e4f38f045f9566fa0
    license='MIT License',
    author='Fernando Elias de Melo Borges', 
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email='fernandoelias.mb@gmail.com',
    url='https://github.com/fmborges2/PC_Classifier/',
    keywords= ['one class classifier', 'data classification', 'principal curves', 'k segments'],
    description=u'One Class Classifier based on Principal Curves in Python',
    packages=['ocpc_py'],
    install_requires=['numpy', 'matplotlib'], )