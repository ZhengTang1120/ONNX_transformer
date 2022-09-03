import scala.io.Source

import ai.onnxruntime.{OnnxTensor, OrtEnvironment, OrtSession}

import java.time.LocalDateTime
import java.time.Duration


import scala.util.parsing.json._


object TestOnnx extends App {

  val ortEnvironment = OrtEnvironment.getEnvironment
  val modelpath1 = "model.onnx"
  val session1 = ortEnvironment.createSession(modelpath1, new OrtSession.SessionOptions)

  println(session1.getOutputInfo)

  val start_time = LocalDateTime.now()

  val jsonString = Source.fromFile("word_ids.json").getLines.mkString
  val parsed = JSON.parseFull(jsonString)


  val words = parsed.get.asInstanceOf[List[Any]].map(l => l.asInstanceOf[List[Any]].map(i => i.asInstanceOf[Number].longValue).toArray)
  val word_input = new java.util.HashMap[String, OnnxTensor]()
  word_input.put("input_ids",  OnnxTensor.createTensor(ortEnvironment, words.toArray))
  val emissionScores = session1.run(word_input).get(0).getValue.asInstanceOf[Array[Array[Array[Float]]]]

  print(emissionScores(0).map(_.mkString(" ")).mkString("\n"))
}