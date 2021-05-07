package com.github.schuettec.ai.neuralnetwork.examples.trafficlights;

import java.util.Arrays;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Perceptron;

public class Main {

  public static class TrafficLights {
    private double[] inputRow = new double[5];

    public TrafficLights red() {
      inputRow[0] = 1;
      inputRow[1] = 0;
      inputRow[2] = 0;
      return this;
    }

    public TrafficLights orange() {
      inputRow[0] = 0;
      inputRow[1] = 1;
      inputRow[2] = 0;
      return this;
    }

    public TrafficLights green() {
      inputRow[0] = 0;
      inputRow[1] = 0;
      inputRow[2] = 1;
      return this;
    }

    public TrafficLights carComing() {
      inputRow[3] = 1;
      return this;
    }

    public TrafficLights walking() {
      inputRow[4] = 1;
      return this;
    }

    public double[] toInput() {
      return inputRow;
    }

    public DataSetRow thenDead() {
      return new DataSetRow(inputRow, new double[] {
          0
      });
    }

    public DataSetRow thenOk() {
      return new DataSetRow(inputRow, new double[] {
          1
      });
    }

    public static TrafficLights create() {
      TrafficLights trafficLights = new TrafficLights();
      return trafficLights;
    }

    public static String toAnswer(double[] answer) {
      if (answer[0] == 0) {
        return "You're dead!";
      } else {
        return "You're fine!";
      }
    }
  }

  public static void main(String[] args) {

    // Ampel Rot, Ampel Gelb, Ampel Grün, Fahrzeug sichtbar

    DataSet trainingSet = new DataSet(5, 1);
    trainingSet.add(TrafficLights.create()
        .red()
        .walking()
        .thenOk());
    trainingSet.add(TrafficLights.create()
        .orange()
        .walking()
        .thenOk());
    trainingSet.add(TrafficLights.create()
        .green()
        .walking()
        .thenOk());
    trainingSet.add(TrafficLights.create()
        .red()
        .carComing()
        .walking()
        .thenDead());
    trainingSet.add(TrafficLights.create()
        .orange()
        .carComing()
        .walking()
        .thenDead());
    trainingSet.add(TrafficLights.create()
        .green()
        .carComing()
        .walking()
        .thenDead());
    trainingSet.add(TrafficLights.create()
        .red()
        .carComing()
        .thenOk());
    trainingSet.add(TrafficLights.create()
        .orange()
        .carComing()
        .thenOk());
    trainingSet.add(TrafficLights.create()
        .green()
        .carComing()
        .thenOk());

    // create perceptron neural network
    NeuralNetwork myPerceptron = new Perceptron(5, 1);
    // learn the training set
    myPerceptron.learn(trainingSet);

    // save trained perceptron
    myPerceptron.save("trafficLights.nnet");
    // load saved neural network
    NeuralNetwork loadedPerceptron = NeuralNetwork.createFromFile("trafficLights.nnet");

    {
      double[] answer = question(loadedPerceptron, TrafficLights.create()
          .red()
          .walking()
          .toInput());

      System.out.println(TrafficLights.toAnswer(answer));
    }
    {
      double[] answer = question(loadedPerceptron, TrafficLights.create()
          .red()
          .carComing()
          .walking()
          .toInput());

      System.out.println(TrafficLights.toAnswer(answer));
    }
    {
      double[] answer = question(loadedPerceptron, TrafficLights.create()
          .red()
          .carComing()
          .toInput());

      System.out.println(TrafficLights.toAnswer(answer));
    }

    // test loaded neural network
    // System.out.println("Testing loaded perceptron");
    // testNeuralNetwork(loadedPerceptron, trainingSet);
    // unknownCaseNeuralNetwork(loadedPerceptron);

  }

  public static double[] question(NeuralNetwork neuralNetwork, double[] input) {
    neuralNetwork.setInput(input);
    neuralNetwork.calculate();
    return neuralNetwork.getOutput();
  }

  public static void testNeuralNetwork(NeuralNetwork neuralNet, DataSet testSet) {
    for (DataSetRow trainingElement : testSet.getRows()) {
      neuralNet.setInput(trainingElement.getInput());
      neuralNet.calculate();
      double[] networkOutput = neuralNet.getOutput();
      System.out.print("Input: " + Arrays.toString(trainingElement.getInput()));
      System.out.println(" Output: " + Arrays.toString(networkOutput));
    }
  }

  public static void unknownCaseNeuralNetwork(NeuralNetwork neuralNet) {
    DataSetRow unknownElement = new DataSetRow();// to print only
    unknownElement.setInput(new double[] {
        1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
    });// to print only
    neuralNet.setInput(new double[] {
        1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
    });
    neuralNet.calculate();
    double[] networkOutput = neuralNet.getOutput();
    System.out.print("unknown case, input: " + Arrays.toString(unknownElement.getInput()));
    System.out.println(" output: " + Arrays.toString(networkOutput));
  }
}
