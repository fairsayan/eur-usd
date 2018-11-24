package EurUsd

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession

import org.apache.log4j.{Level, Logger}

object Training {

  def main(args: Array[String]): Unit = {

    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    val spark = SparkSession
      .builder
      .appName("Training")
      .getOrCreate()

    val data = spark.read.format("libsvm")
      .load("data.txt")

    data.show()

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val lr = new LinearRegression()
      .setMaxIter(10)

    val lrModel = lr.fit(trainingData)
    val trainingSummary = lrModel.summary

    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")

    val predictions = lrModel.transform(testData)
    predictions.select("prediction", "label", "features").show(5)

    lrModel.write.overwrite().save("model")

    spark.stop()
  }
}