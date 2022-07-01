package ejemplos

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

object AAregresionLOG {
  def main(args: Array[String]): Unit = {
    //Creando el contexto del Servidor
    Logger.getLogger("org").setLevel(Level.OFF)
    val sc = new SparkContext("local",
      "Ejemplo01Base",
      System.getenv("SPARK_HOME"))
    sc.setLogLevel("ERROR")
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("CargaJSON")
      .getOrCreate()
    import spark.implicits._
    case class Obs(clas: Double, thickness: Double, size: Double, shape: Double, madh: Double, epsize: Double, bnuc: Double, bchrom: Double, nNuc: Double, mit: Double)
    // function to create a Obs class from an Array of Double.Class Malignant 4 is changed to 1
    def parseObs(line: Array[Double]): Obs = {
      Obs(
        if (line(9) == 4.0) 1 else 0, line(0), line(1), line(2), line(3), line(4), line(5), line(6), line(7), line(8)
      )
    }
    // function to transform an RDD of Strings into an RDD of Double, filter lines with ?, remove first column
    def parseRDD(rdd: RDD[String]): RDD[Array[Double]] = {
      rdd.map(_.split(",")).filter(_(6) != "?").map(_.drop(1)).map(_.map(_.toDouble))
    }
    val csv = sc.textFile("resources/TitanicSucio.csv").cache()


    var miDF=spark.read
      .option("inferSchema","true")
      .option("header", "true")
      .option("delimiter", ",")
      .csv("resources/TitanicSucio.csv")
      .cache()
    miDF.printSchema()
    var obsDF=miDF.filter(_(6) != "?").drop("id").cache()
    obsDF.printSchema()
    obsDF.head(1).foreach(println)
    /*

    val soloDatosDF=filtrado.drop("class").cache()
    soloDatosDF.printSchema()
    soloDatosDF.show()

    var cambiadoDF=soloDatosDF.withColumn("clump_thickness",$"clump_thickness".cast(DoubleType))
    cambiadoDF=soloDatosDF.withColumn("unif_cell_size",$"unif_cell_size".cast(DoubleType))
    cambiadoDF=soloDatosDF.withColumn("unif_cell_shape",$"unif_cell_shape".cast(DoubleType))
    cambiadoDF.printSchema()
    cambiadoDF=soloDatosDF.withColumn("marg_adhesion",$"marg_adhesion".cast(DoubleType))
    cambiadoDF=soloDatosDF.withColumn("single_epith_cell_size",$"single_epith_cell_size".cast(DoubleType))
    cambiadoDF=soloDatosDF.withColumn("bare_nuclei",$"bare_nuclei".cast(DoubleType))
    cambiadoDF=soloDatosDF.withColumn("bland_chrom",$"bland_chrom".cast(DoubleType))
    cambiadoDF=soloDatosDF.withColumn("norm_nucleoli",$"norm_nucleoli".cast(DoubleType))
    cambiadoDF=soloDatosDF.withColumn("mitoses",$"mitoses".cast(DoubleType))
    cambiadoDF.printSchema()
    */
    obsDF=obsDF.withColumn("fare",$"fare".cast(DoubleType))
    //val obsDF = parseRDD(csv).map(parseObs).toDF()
    obsDF.createOrReplaceGlobalTempView("obs")
    var obs1 = obsDF.na.drop("any")
    obs1.printSchema
    obs1.show
    //obsDF.describe("thickness").show
    //define the feature columns to put in the feature vector""
    val featureCols = Array("pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "class", "who", "adult_male", "deck", "embark_town", "alone")

    //set the input and output column names
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    //return a dataframe with all of the  feature columns in  a vector column
    val df2 = assembler.transform(obs1)
    // the transform method produced a new column: features.
    df2.show
    //  Create a label column with the StringIndexer
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("label1")
    val df3 = labelIndexer.fit(df2).transform(df2)
    // the  transform method produced a new column: label.
    df3.show
    //  split the dataframe into training and test data
    val splitSeed = 5043
    val Array(trainingData, testData) = df3.randomSplit(Array(0.7, 0.3), splitSeed)

    // create the classifier,  set parameters for training
    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    //  use logistic regression to train (fit) the model with the training data
    val model = lr.fit(trainingData)

    // Print the coefficients and intercept for logistic regression
    println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")

    // run the  model on test features to get predictions**
    val predictions = model.transform(testData)
    //As you can see, the previous model transform produced a new columns: rawPrediction, probablity and prediction.**
    predictions.show(500)

    //A common metric used for logistic regression is area under the ROC curve (AUC). We can use the BinaryClasssificationEvaluator to obtain the AUC
    // create an Evaluator for binary classification, which expects two input columns: rawPrediction and label.
    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label1").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")
    // Evaluates predictions and returns a scalar metric areaUnderROC(larger is better).
    val accuracy = evaluator.evaluate(predictions)

    // Calculate Metrics
    val lp = predictions.select( "label1", "prediction")
    val counttotal = predictions.count()
    val correct = lp.filter($"label1" === $"prediction").count()
    val wrong = lp.filter(not($"label1" === $"prediction")).count()
    val truep = lp.filter($"prediction" === 0.0).filter($"label1" === $"prediction").count()
    val falseN = lp.filter($"prediction" === 0.0).filter(not($"label1" === $"prediction")).count()
    val falseP = lp.filter($"prediction" === 1.0).filter(not($"label1" === $"prediction")).count()
    val ratioWrong=wrong.toDouble/counttotal.toDouble
    val ratioCorrect=correct.toDouble/counttotal.toDouble
    println(ratioWrong)
    println(ratioCorrect)
    // use MLlib to evaluate, convert DF to RDD
    /*
    val  predictionAndLabels =predictions.select("rawPrediction", "label").rdd.map(x => (x(0).asInstanceOf[DenseVector](1), x(1).asInstanceOf[Double]))
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    println("area under the precision-recall curve: " + metrics.areaUnderPR)
    println("area under the receiver operating characteristic (ROC) curve : " + metrics.areaUnderROC)
    */
  }


}

