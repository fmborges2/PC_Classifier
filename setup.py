from setuptools import setup

with open("README.md", "r") as arq:
    readme = arq.read()

setup(name='kseg_py',
    version='0.0.4',
    license='MIT License',
    author='Fernando Elias de Melo Borges',
    long_description=readme,
    long_description_content_type="text/markdown",
    author_email='fernandoelias.mb@gmail.com',
    url='https://github.com/fmborges2/PC_Classifier/',
    keywords='principal curves',
    description=u'K-segments for Principal Curves Extraction in Python',
    packages=['kseg_py'],
    install_requires=['numpy', 'matplotlib'], )