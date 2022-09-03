ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.11.1"

lazy val root = (project in file("."))
  .settings(
    name := "ONNX_transformer"
  )
libraryDependencies ++= {
  Seq(
    "com.microsoft.onnxruntime"  % "onnxruntime" % "1.8.1",
    "org.scala-lang.modules" %% "scala-parser-combinators" % "1.0.4",
  )
}