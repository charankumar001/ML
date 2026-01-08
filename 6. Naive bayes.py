{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOxzUl6RNteg9tzyuoQT/tZ",
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
      "source": [
        "from sklearn.datasets import load_iris\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.naive_bayes import GaussianNB\n",
        "from sklearn.metrics import confusion_matrix, accuracy_score\n",
        "X, y = load_iris(return_X_y=True)\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "naive_bayes_classifier = GaussianNB()\n",
        "naive_bayes_classifier.fit(X_train, y_train)\n",
        "y_pred = naive_bayes_classifier.predict(X_test)\n",
        "conf_matrix = confusion_matrix(y_test, y_pred)\n",
        "print(\"Confusion Matrix:\")\n",
        "print(conf_matrix)\n",
        "accuracy = accuracy_score(y_test, y_pred) * 100\n",
        "print(f\"Accuracy: {accuracy:.2f}%\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "34BmCl1yFnai",
        "outputId": "f4883e00-ab3a-46d5-e017-e24767e499d8"
      },
      "execution_count": 2,
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
      ]
    }
  ]
}