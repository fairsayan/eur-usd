# eur-usd
Example how to use Apache Spark MLLib in SCALA to predict exchange EUR-USD
IntelliJ project files

## usage
- sbt package: to compile the project *or* select "sbt shell" from the bottom of the screen and write the "package" command
- spark-submit --class EurUsd.Training target/.../eur-usd....jar (creates "model" directory)
- spark-submit --class EurUsd.Wizard target/.../eur-usd....jar
