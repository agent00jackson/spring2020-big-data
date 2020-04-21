import org.apache.spark.sql._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.evaluation._
import spark.implicits._

import com.iresium.ml.SMOTE

def loadData() : DataFrame = {
    val rawDf = {spark.read.format("csv")
	.option("header","true")
    .option("inferSchema", "true")
	.load("/files/creditcard.csv")
    }

    val vCols = rawDf.drop("Class").columns

    val assembler = {new VectorAssembler()
        .setInputCols(vCols)
        .setOutputCol("features")
    }

    return assembler.transform(rawDf).drop(vCols:_*)
}

def trainLRModel(trainData: DataFrame, iterations: Int, tol: Double) : LogisticRegressionModel = {
    val lr = {new LogisticRegression()
        .setMaxIter(iterations)
        .setRegParam(0.3)
        .setElasticNetParam(0.8)
        .setTol(tol)
        .setLabelCol("Class")
        .setFeaturesCol("features")
    }
    return lr.fit(trainData)
}

def RunMetrics(theModel: LogisticRegressionModel, validData: DataFrame){
    val trainingSummary = theModel.binarySummary
    println("Training:")
    println(s"\tAccuracy: ${trainingSummary.accuracy}")
    println(s"\tAUROC: ${trainingSummary.areaUnderROC}")
    println(s"\tIterations: ${trainingSummary.totalIterations}")
    trainingSummary.fMeasureByLabel.zipWithIndex.foreach { case (f, label) =>
      println(s"\tF1-Score($label) = $f")
    }
    trainingSummary.precisionByLabel.zipWithIndex.foreach { case (f, label) =>
      println(s"\tPrecision($label) = $f")
    }
    trainingSummary.recallByLabel.zipWithIndex.foreach { case (f, label) =>
      println(s"\tRecall($label) = $f")
    }

    println("Validation:")
    val validSummary = theModel.evaluate(validData).asBinary
    println(s"\tAccuracy: ${validSummary.accuracy}")
    println(s"\tAUROC: ${validSummary.areaUnderROC}")
    validSummary.fMeasureByLabel.zipWithIndex.foreach { case (f, label) =>
      println(s"\tF1-Score($label) = $f")
    }
    validSummary.precisionByLabel.zipWithIndex.foreach { case (f, label) =>
      println(s"\tPrecision($label) = $f")
    }
    validSummary.recallByLabel.zipWithIndex.foreach { case (f, label) =>
      println(s"\tRecall($label) = $f")
    }
}

def undersample(theData: DataFrame, ratio: Double) : DataFrame = {
    val pos = theData.filter($"Class" === 1)
    val neg = theData.filter($"Class" === 0)
    val p_count = pos.count()
    val n_count = neg.count()
    val frac = (p_count * ratio) / n_count
    val sampled = neg.sample(false, frac, 11L)
    return sampled.union(pos)
}

def runSmote(theData: DataFrame) : DataFrame = {
    val smote = {
        new SMOTE()
        .setfeatureCol("features")
        .setlabelCol("Class")
        .setbucketLength(100)
    }

    val smoteModel = smote.fit(theData)
    return smoteModel.transform(theData)
}

def trainRFModel(theData: DataFrame) : RandomForestClassificationModel = {
    val rf = {new RandomForestClassifier()
        .setLabelCol("Class")
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setSeed(11L)
    }
    return rf.fit(theData)
}

def RunRFMetrics(theModel: RandomForestClassificationModel, theData: DataFrame){
    val eval = theModel.transform(theData)

    val tp = eval.filter($"prediction" === 1.0 && $"Class" === 1.0).count()
    val fn = eval.filter($"prediction" === 0.0 && $"Class" === 1.0).count()
    val recall = tp.toFloat / (tp+fn)

    val fp = eval.filter($"prediction" === 1.0 && $"Class" === 0.0).count()
    val precision = tp.toFloat / (tp+fp)

    val tn = eval.filter($"prediction" === 0.0 && $"Class" === 0.0).count()
    val accuracy = (tp.toFloat + tn.toFloat) / eval.count()

    val f1 = 2.0 * (precision * recall) / (precision + recall)
    println(s"\tAccuracy: $accuracy")
    println(s"\tPrecision: $precision")
    println(s"\tRecall: $recall")
    println(s"\tF1: $f1")

    //val toArr: Any => Array[Double] = _.asInstanceOf[org.apache.spark.ml.linalg.Vector].toArray
    //val toArrUDF = udf(toArr)
//
    //val fixed = eval.withColumn("a_probability", toArrUDF($"probability"))
    //val multiclassEval = {
    //    new MultilabelClassificationEvaluator()
    //    .setLabelCol("Class")
    //    .setPredictionCol("prediction")
    //}
    val binEval = {
        new BinaryClassificationEvaluator()
        .setLabelCol("Class")
        .setRawPredictionCol("rawPrediction")
    }
    //def McMetric(metric: String) = multiclassEval.setMetricName(metric).evaluate(fixed)
    //println("\tf1 Measure: " + McMetric("f1Measure"))
//
    //val precision = McMetric("precisionByLabel")
    //println(s"\tprecision: $precision")
    //
    //val recall = McMetric("recallByLabel")
    //println(s"\trecall: $recall")

    def printBinMetric(metric: String) = println(s"\t$metric: " + binEval.setMetricName(metric).evaluate(eval).toString)
    printBinMetric("areaUnderROC")
}

val data = loadData()
val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
val training = splits(0)
val validation = splits(1)

val smoted = runSmote(training)
val trainResample = undersample(smoted, 2)
val undersamp = undersample(training, .97)

val smoteModel = trainLRModel(trainResample, 10, 1e-6)
val undModel = trainLRModel(undersamp, 9, 1e-6)
val noresampleModel = trainLRModel(training, 10, 1e-6)

val rfModel = trainRFModel(training)
val rf_smote_model = trainRFModel(smoted)
val rf_und_model = trainRFModel(undersamp)

println("---------------")
println(" Random Forest ")
println("---------------")
println("Smote + Undersampling:")
println("Training:")
RunRFMetrics(rf_smote_model, smoted)
println("Validation:")
RunRFMetrics(rf_smote_model, validation)

println("Undersampling:")
println("Training:")
RunRFMetrics(rf_und_model, undersamp)
println("Validation:")
RunRFMetrics(rf_und_model, validation)

println("No Resampling:")
println("Training:")
RunRFMetrics(rfModel, training)
println("Validation:")
RunRFMetrics(rfModel, validation)

println("")
println("---------------------")
println(" Logistic Regression ")
println("---------------------")
println("Smote + Undersampling:")
RunMetrics(smoteModel, validation)

println("Undersampling:")
RunMetrics(undModel, validation)

println("No Resampling:")
RunMetrics(noresampleModel, validation)