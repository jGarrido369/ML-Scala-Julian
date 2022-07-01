package ejemplos

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession



object ArbolProySIMPLE {


  def main(args: Array[String]): Unit = {
    //Reducir el número de LOG
    Logger.getLogger("org").setLevel(Level.OFF)
    //Creando el contexto del Servidor
    val sc = new SparkContext("local","ArbolProy", System.getenv("SPARK_HOME"))
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("CargaJSON")
      .config("log4j.rootCategory", "ERROR, console")
      .getOrCreate()
    var df = spark.read.format("csv")
      .option("header", "true")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .load("resources/TitanicSucio.csv")
    /*
    df = df.withColumnRenamed("_c0", "sepalLength")
    df = df.withColumnRenamed("_c1", "sepalWidth")
    df = df.withColumnRenamed("_c2", "petalLength")
    df = df.withColumnRenamed("_c3", "petalWidth")
    df = df.withColumnRenamed("_c4", "labels")

    // identify the feature colunms

    */
    df.printSchema()
    val df2 = df.drop("alive","deck")

    //se quita alive por redundancia obteniendo 0.872 y se quita deck por tener muchos nulls 0.877
    //val df3 = df2.na.drop("any")
    //0.8010213299411906 con "pclass","sex","fare","class"
    //0.8242517025508989 con "pclass","sex","fare","class","sibsp","fare","class","who","adult_male","alone"
    //0.827595385980479 con "pclass","sex","class","parch","class","sibsp","fare","who","adult_male","alone","embarked","embark_town"


    val inputColumns = Array("pclass","sex","class","parch","class","sibsp","fare","who","adult_male","alone","embarked","embark_town")
    val assembler = new VectorAssembler().setInputCols(inputColumns).setOutputCol("features")

    val featureSet = assembler.transform(df2)

    // split data random in trainingset (70%) and testset (30%)
    val seed = 5043
    val trainingAndTestSet = featureSet.randomSplit(Array[Double](0.69, 0.31), seed)
    //con esta configuracion nos sale el mejor accuracy 0.8287636407541525
    val trainingSet = trainingAndTestSet(0)
    val testSet = trainingAndTestSet(1)

    // train the algorithm based on a Random Forest Classification Algorithm with default values// train the algorithm based on a Random Forest Classification Algorithm with default values

    val randomForestClassifier = new RandomForestClassifier().setSeed(seed)
    //randomForestClassifier.setMaxDepth(4)
    val model = randomForestClassifier.fit(trainingSet)
    // test the model against the test set       
    val predictions = model.transform(testSet)

    // evaluate the model
    val evaluator = new MulticlassClassificationEvaluator()
    df2.show()
    System.out.println("accuracy: " + evaluator.evaluate(predictions))
    //df2.show(714)
    //println(df3.count())

    //df.filter(df(colName).isNull || df(colName) === "" || df(colName).isNaN).count()
    /*
    val dataMapped=data.map( _.toDouble )
    //Dividiendo datos en training y test
    val Array(training, test) = data.randomSplit(Array[Double](0.7, 0.3), 18)
    training.show()
    */
    /*
    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(training)
    // Print the coefficients and intercept for linear regression
      println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
    */
    /*


    case class Medidas(petalLength: Float,petalWidth: Float,sepalLength: Float,sepalWidth: Float, feature:String)

    var df = spark.read.format("csv").option("delimiter", ",")
      .load("resources/iris-multiclass.csv").toDF()
    df.show()
    df = df.withColumnRenamed("_c0", "sepalLength")
    df = df.withColumnRenamed("_c1", "sepalWidth")
    df = df.withColumnRenamed("_c2", "petalLength")
    df = df.withColumnRenamed("_c3", "petalWidth")
    df = df.withColumnRenamed("_c4", "features")
    df.show()





    var onlyData = df.drop("features").cache()
    onlyData.show()
    var primero=onlyData.first()
    println(primero)
    val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel = lr.fit(df)
      //.dropRight(1).map(_.toDouble))
       */
    /*
    val numClusters = 3
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    val WSSSE = clusters.computeCost(parsedData)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    sc.stop()
    */
  }

}