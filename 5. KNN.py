{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyO91bs4S1X2SvrI/ijW+a/y",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/charankumar001/ML/blob/main/5.%20KNN.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "jQ7pe1X8D_zv",
        "outputId": "cf0576d2-f121-4260-f980-bceff68b56c9"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 83.33%\n"
          ]
        }
      ],
      "source": [
        "from sklearn.datasets import load_iris\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.metrics import accuracy_score\n",
        "import numpy as np\n",
        "iris = load_iris()\n",
        "X, y = iris.data, iris.target\n",
        "np.random.seed(42)\n",
        "noise = np.random.choice([0, 1], size=len(y), p=[0.1, 0.9])\n",
        "y_noisy = (y + noise) % 3\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=0.2, random_state=42)\n",
        "knn_classifier = KNeighborsClassifier(n_neighbors=3)\n",
        "knn_classifier.fit(X_train, y_train)\n",
        "y_pred = knn_classifier.predict(X_test)\n",
        "accuracy = accuracy_score(y_test, y_pred) * 100\n",
        "print(f\"Accuracy: {accuracy:.2f}%\")"
      ]
    }
  ]
}