package EurUsd

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.sql.SparkSession

import org.apache.log4j.{Level, Logger}

object Wizard {
    def main(args: Array[String]) {
      val spark = SparkSession
        .builder
        .appName("Wizard")
        .getOrCreate()

      val rootLogger = Logger.getRootLogger()
      rootLogger.setLevel(Level.ERROR)

      val data = spark.read.format("libsvm").load("oggi.txt")
      val lrModel = LinearRegressionModel.load("model")

      println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

      val predictions = lrModel.transform(data)
      predictions.select("prediction", "features").show()
    }
}

