from setuptools import setup

with open("README.md", "r") as arq:
    readme = arq.read()

setup(name='ocpc_py',
    version='0.1.4',
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