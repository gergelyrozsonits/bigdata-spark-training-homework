package com.epam.training.spark.core

import java.time.LocalDate
import java.time.format.DateTimeFormatter

import com.epam.training.spark.core.domain.Climate
import com.epam.training.spark.core.domain.ClimateTypes.{PrecipitationAmount, SunshineHours, Temperature}
import com.epam.training.spark.core.domain.PrecipitationType.Precipitation
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object Homework {
  val DELIMITER = ";"
  val RAW_BUDAPEST_DATA = "data/budapest_daily_1901-2010.csv"
  val OUTPUT_DUR = "output"

  def main(args: Array[String]): Unit = {
    val sparkConf: SparkConf = new SparkConf()
      .setAppName("EPAM BigData training Spark Core homework")
      .setIfMissing("spark.master", "local[2]")
      .setIfMissing("spark.sql.shuffle.partitions", "10")
    val sc = new SparkContext(sparkConf)

    processData(sc)

    sc.stop()

  }

  def processData(sc: SparkContext): Unit = {

    /**
      * Task 1
      * Read raw data from provided file, remove header, split rows by delimiter
      */
    val rawData: RDD[List[String]] = getRawDataWithoutHeader(sc, Homework.RAW_BUDAPEST_DATA)

    /**
      * Task 2
      * Find errors or missing values in the data
      */
    val errors: List[Int] = findErrors(rawData)
    println(errors)

    /**
      * Task 3
      * Map raw data to Climate type
      */
    val climateRdd: RDD[Climate] = mapToClimate(rawData)

    /**
      * Task 4
      * List average temperature for a given day in every year
      */
    val averageTemeperatureRdd: RDD[Double] = averageTemperature(climateRdd, 1, 2)

    /**
      * Task 5
      * Predict temperature based on mean temperature for every year including 1 day before and after
      * For the given month 1 and day 2 (2nd January) include days 1st January and 3rd January in the calculation
      */
    val predictedTemperature: Double = predictTemperature(climateRdd, 1, 2)
    println(s"Predicted temperature: $predictedTemperature")

  }

  def getRawDataWithoutHeader(sc: SparkContext, rawDataPath: String): RDD[List[String]] = {
    val csv = sc.textFile(rawDataPath);
    val data = csv.map(line => line.split(";", -1).toList);
    val first = data.first();
    data.filter(item => !item.equals(first));
  }

  def findErrors(rawData: RDD[List[String]]): List[Int] = {
    rawData
      .map(_.map(word  => if(word.isEmpty) 1 else 0))
      .reduce((a, b) => a.zipWithIndex.map{case (value, index) => value + b(index)})
  }

  def mapToClimate(rawData: RDD[List[String]]): RDD[Climate] = {
    rawData.map(raw => new Climate(
      LocalDate.parse(raw(0), DateTimeFormatter.ofPattern("yyyy-MM-dd")),
      Temperature.apply(raw.lift(1).get),
      Temperature.apply(raw.lift(2).get),
      Temperature.apply(raw.lift(3).get),
      PrecipitationAmount.apply(raw.lift(4).get),
      Precipitation.apply(raw.lift(5).get),
      SunshineHours.apply(raw.lift(6).get)
    ))
  }

  def averageTemperature(climateData: RDD[Climate], month: Int, dayOfMonth: Int): RDD[Double] = {
    climateData
      .filter(climate => climate.observationDate.getMonthValue.equals(month) && climate.observationDate.getDayOfMonth.equals(dayOfMonth))
      .map(climate => climate.meanTemperature.value);
  }

  def predictTemperature(climateData: RDD[Climate], month: Int, dayOfMonth: Int): Double = {
    val prev = climateData
      .filter(climate => LocalDate.of(climate.observationDate.getYear(), month, dayOfMonth)
        .minusDays(1)
        .equals(climate.observationDate))
      .map(climate => (climate.observationDate.getYear, climate.meanTemperature.value));
    val next = climateData
      .filter(climate => LocalDate.of(climate.observationDate.getYear(), month, dayOfMonth)
        .plusDays(1)
        .equals(climate.observationDate))
      .map(climate => (climate.observationDate.getYear, climate.meanTemperature.value));
    val curr = climateData
      .filter(climate => LocalDate.of(climate.observationDate.getYear(), month, dayOfMonth)
        .equals(climate.observationDate))
      .map(climate => (climate.observationDate.getYear, climate.meanTemperature.value));
    curr
      .join(prev)
      .join(next)
      .map { case (year, ((a, b), c)) => (a + b + c) / 3 }.mean()
  }
}


