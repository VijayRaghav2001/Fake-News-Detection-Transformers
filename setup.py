from setuptools import setup, find_packages

setup(
    name="fake_news_detection",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.12.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "tensorboard>=2.7.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pyyaml>=5.4.0",
    ],
    entry_points={
        "console_scripts": [
            "fnd-train=fake_news_detection.train:main",
            "fnd-predict=fake_news_detection.predict:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Fake News Detection with Transformer Models",
    keywords="fake news detection transformers nlp",
    python_requires=">=3.7",
)