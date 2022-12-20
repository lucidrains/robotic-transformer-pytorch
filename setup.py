from setuptools import setup, find_packages

setup(
  name = 'robotic-transformer-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.12',
  license='MIT',
  description = 'Robotic Transformer - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/robotic-transformer-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'robotics'
  ],
  install_requires=[
    'classifier-free-guidance-pytorch>=0.0.23',
    'einops>=0.6',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
