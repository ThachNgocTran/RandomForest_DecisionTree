package com.mycompany

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j.{Level, Logger}
import java.nio.file.{Files, Paths}

import org.apache.spark.mllib.linalg._
import org.apache.commons.lang3.exception.ExceptionUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest}
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD

object App {

  def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics ={
    val predictionAndLabels = data.map(x => (model.predict(x.features), x.label))
    new MulticlassMetrics(predictionAndLabels)
  }

  def classProbabilities(data: RDD[LabeledPoint]): Array[Double] ={
    val countCategories = data.map(_.label).countByValue()
    val probs = countCategories.toArray.sortBy(_._1).map(_._2)
    probs.map(_.toDouble / probs.sum) // force to get fraction
  }

  def main(args : Array[String]) {

    println("*** Start ***")

    // Comment out when errors happen.
    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);

    val DATA_PATH = "./data/covtype.data"

    try{

      val conf = new SparkConf().setMaster("local[*]").setAppName("Max Price")
      val sc = new SparkContext(conf)

      if (!Files.exists(Paths.get(DATA_PATH))){
        throw new Exception("[%s] not found.".format(DATA_PATH))
      }


      //////////////////////////////////////////////////////////////////////////////////////////////
      // DATA PROCESSING
      val rawData = sc.textFile(DATA_PATH)
      // Convert raw data to LabeledPoint
      val data = rawData.map(line => {
        /* Original approach: keep one-hot encoding.
        val values = line.split(",").map(_.toDouble)
        val features = Vectors.dense(values.init)
        val label = values.last - 1
        LabeledPoint(label, features)
        */
        val values = line.split(",").map(_.toDouble)
        val wilderness = values.slice(10, 14).indexOf(1.0).toDouble
        val soil = values.slice(14, 54).indexOf(1.0).toDouble
        val features = Vectors.dense(values.slice(0, 10) :+ wilderness :+ soil)
        val label = values.last - 1
        LabeledPoint(label, features)
      })

      // Split the raw data into Training Set, Validation Set, Test Set.
      val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
      trainData.cache()
      cvData.cache()
      testData.cache()


      //////////////////////////////////////////////////////////////////////////////////////////////
      // DECISION TREE

      // Train a Decision Tree against Training Set.
      /* Original approach: keep one-hot encoding.
      val model = DecisionTree.trainClassifier(trainData, 7, Map[Int, Int](), "gini", 4, 100)
      */
      val model = DecisionTree.trainClassifier(trainData, 7, Map[Int, Int](10 -> 4, 11 -> 40), "gini", 4, 100)

      val metrics = getMetrics(model, cvData)

      println("Confusion Matrix:\n%s\n".format(metrics.confusionMatrix.toString()))
      println("Accuracy:\n%s\n".format(metrics.accuracy))

      println("Precision/Recall for each target value (one vs. all):")
      (0 until 7).map(x => println(metrics.precision(x), metrics.recall(x)))

      // Calculate the weighted guessing
      val trainProbabilities = classProbabilities(trainData)
      val cvProbabilities = classProbabilities(cvData)
      val weightedGuessing = trainProbabilities.zip(cvProbabilities).map({ case (a, b) => a * b}).sum

      println("\nWeighted Guessing: %f\n".format(weightedGuessing))

      // Tuning Decision Tree
      val evaluations = for(impurity <- Array("gini", "entropy");
                            depth <- Array(10, 20, 30);
                            bins <- Array(40, 300))
                        yield {
                          /* Original approach: keep one-hot encoding.
                          val model = DecisionTree.trainClassifier(trainData, 7, Map[Int, Int](), impurity, depth, bins)
                          */
                          val model = DecisionTree.trainClassifier(trainData, 7, Map[Int, Int](10 -> 4, 11 -> 40), impurity, depth, bins)
                          val predictionAndLabels = cvData.map(x => (model.predict(x.features), x.label))
                          val accuracy = new MulticlassMetrics(predictionAndLabels).accuracy
                          ((impurity, depth, bins), accuracy)
                        }

      println("Searching for hyperparameters:")
      evaluations.sortBy(_._2).reverse.foreach(println)

      // With one-hot encoding, (entropy,20,300) achieves the highest accuracy on Validation Set
      // Now use these parameters to predict Test Set
      // Note: Union() because MORE DATA, BETTER!
      /* Original approach: keep one-hot encoding.
      val modelFinal = DecisionTree.trainClassifier(trainData.union(cvData), 7, Map[Int, Int](), "entropy", 20, 300)
      */
      // Remove one-hot encoding, with finer grid, (entropy,30,300) achieved 94%
      val modelFinal = DecisionTree.trainClassifier(trainData.union(cvData), 7, Map[Int, Int](10 -> 4, 11 -> 40), "entropy", 30, 300)

      val predictionAndValuesFinal = testData.map(x => (modelFinal.predict(x.features), x.label))
      val metricsFinal = new MulticlassMetrics(predictionAndValuesFinal)

      // With one-hot encoding, 90% accuracy.
      /* Original approach: keep one-hot encoding.
      println("\nWith hyperparams (entropy,20,300), got accuracy: %f\n".format(metricsFinal.accuracy))
       */
      // Remove one-hot encoding, 94% accuracy on Test Set.
      println("\nWith hyperparams (entropy, 30, 300), got accuracy: %f\n".format(metricsFinal.accuracy))


      //////////////////////////////////////////////////////////////////////////////////////////////
      // RANDOM FOREST
      println("Using Random Forest...")

      val modelRF = RandomForest.trainClassifier(trainData.union(cvData), 7, Map[Int, Int](10->4, 11->40), 20, "auto", "entropy", 30, 300);
      val predictionAndLabelsRF = testData.map(x => (modelRF.predict(x.features), x.label))
      val metricsRF = new MulticlassMetrics(predictionAndLabelsRF)

      // Got 0.964704
      println("With hyperparams (20, auto, entropy, 30, 300), got accuracy: %f".format(metricsRF.accuracy))

      println("*** Done! ***")
    }
    catch {
      case e: Throwable => println(ExceptionUtils.getStackFrames(e))
    }
  }
}

/*
Sample Output:

*** Start ***
Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
SLF4J: Class path contains multiple SLF4J bindings.
SLF4J: Found binding in [jar:file:/C:/spark/jars/slf4j-log4j12-1.7.16.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: Found binding in [jar:file:/C:/Users/iRobot/.m2/repository/org/slf4j/slf4j-log4j12/1.7.16/slf4j-log4j12-1.7.16.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: See http://www.slf4j.org/codes.html#multiple_bindings for an explanation.
SLF4J: Actual binding is of type [org.slf4j.impl.Log4jLoggerFactory]
Confusion Matrix:
13580.0  7382.0   0.0     0.0  0.0  0.0  67.0
5058.0   23000.0  318.0   0.0  3.0  0.0  11.0
0.0      1085.0   2561.0  0.0  0.0  0.0  0.0
0.0      0.0      251.0   0.0  0.0  0.0  0.0
0.0      918.0    0.0     0.0  9.0  0.0  0.0
0.0      484.0    1263.0  0.0  0.0  0.0  0.0
1562.0   82.0     0.0     0.0  0.0  0.0  466.0

Accuracy:
0.6818588640275387

Precision/Recall for each target value (one vs. all):
(0.6722772277227723,0.6457748823053878)
(0.6980061303147097,0.8101444170482565)
(0.5829729114500342,0.7024136039495338)
(0.0,0.0)
(0.75,0.009708737864077669)
(0.0,0.0)
(0.8566176470588235,0.22085308056872038)

Weighted Guessing: 0.376455

Searching for hyperparameters:
((entropy,30,300),0.9404819277108434)
((entropy,30,40),0.9377624784853701)
((gini,30,300),0.936764199655766)
((gini,30,40),0.9334595524956971)
((gini,20,300),0.9259896729776248)
((entropy,20,40),0.9244061962134251)
((gini,20,40),0.9218588640275387)
((entropy,20,300),0.921342512908778)
((gini,10,300),0.7925645438898451)
((gini,10,40),0.7871600688468159)
((entropy,10,40),0.7799827882960413)
((entropy,10,300),0.7751290877796901)

With hyperparams (entropy, 30, 300), got accuracy: 0.946194

Using Random Forest...
With hyperparams (20, auto, entropy, 30, 300), got accuracy: 0.964107
*** Done! ***

 */