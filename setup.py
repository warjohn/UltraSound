from setuptools import setup, find_packages

setup(
    name="segmentUSD",  # Название пакета
    version="0.1.1",  # Версия
    description="A library for processing ultrasound DICOM images",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author="Menar",
    author_email="johnvoronina@gmail.com",
    url="https://github.com/warjohn/UltraSound.git",  # Ссылка на GitHub (если есть)
    packages=find_packages(),  # Автоматически находит все пакеты в проекте
    install_requires=[
        "opencv-python",
        "pydicom",
        "numpy",
        "pynrrd"
    ],
    python_requires=">=3.8"
)
