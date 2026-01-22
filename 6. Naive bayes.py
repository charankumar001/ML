{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMRq6jm2cmLeaIIuh+hteg7",
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
        "<a href=\"https://colab.research.google.com/github/charankumar001/ML/blob/main/6.%20Naive%20bayes.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MUReaYCqgygi",
        "outputId": "317dab4c-41c0-4ef8-b46a-f51753f02d50"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Confusion Matrix:\n",
            "[[10  0  0]\n",
            " [ 0  9  0]\n",
            " [ 0  0 11]]\n",
            "Accuracy: 100.00%\n"
          ]
        }
      ],
      "source": [
        "from sklearn.datasets import load_iris\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.naive_bayes import GaussianNB\n",
        "from sklearn.metrics import confusion_matrix, accuracy_score\n",
        "\n",
        "X, y = load_iris(return_X_y=True)\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "naive_bayes_classifier = GaussianNB()\n",
        "\n",
        "naive_bayes_classifier.fit(X_train, y_train)\n",
        "\n",
        "y_pred = naive_bayes_classifier.predict(X_test)\n",
        "\n",
        "conf_matrix = confusion_matrix(y_test, y_pred)\n",
        "print(\"Confusion Matrix:\")\n",
        "print(conf_matrix)\n",
        "\n",
        "accuracy = accuracy_score(y_test, y_pred) * 100\n",
        "print(f\"Accuracy: {accuracy:.2f}%\")"
      ]
    }
  ]
}