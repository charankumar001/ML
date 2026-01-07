{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMp8eqZsA7QOwf0pdBNlQKW",
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
        "<a href=\"https://colab.research.google.com/github/charankumar001/ML/blob/main/1.%20FIND-S%20algorithm.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def find_s_algorithm(training_data):\n",
        "    hypothesis = training_data[0][:-1]\n",
        "\n",
        "    for example in training_data:\n",
        "        if example[-1] == 'Yes':\n",
        "            for i in range(len(hypothesis)):\n",
        "\n",
        "                if example[i] != hypothesis[i]:\n",
        "                    hypothesis[i] = '?'\n",
        "\n",
        "    return hypothesis\n",
        "training_data = [\n",
        "    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],\n",
        "    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],\n",
        "    ['Rainy', 'Cold', 'High', 'Weak', 'Cool', 'Change', 'No'],\n",
        "    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']\n",
        "]\n",
        "result_hypothesis = find_s_algorithm(training_data)\n",
        "print(\"Result Hypothesis:\", result_hypothesis)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MyqY1K12eb_c",
        "outputId": "0f45980b-15b0-4c66-99ba-93a6aab33ac6"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Result Hypothesis: ['Sunny', 'Warm', '?', 'Strong', '?', '?']\n"
          ]
        }
      ]
    }
  ]
}