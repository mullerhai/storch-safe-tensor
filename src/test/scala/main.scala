import torch.utils.safetensors.SafetensorsViewer
import torch.utils.safetensors.encoder.SafetensorsBuilder

import java.io.File
import java.nio.FloatBuffer


def readTensors(file: File = new File("example.safetensors")): Unit = {
    val file = new File("example.safetensors")
    val viewer = SafetensorsViewer.load(file)

    // Get a ByteBuffer
    val longByteBuffer = viewer.getByteBuffer("long_tensor")

    // Get a LongBuffer
    val longBuffer = viewer.getLongBuffer("long_tensor")

    // Get a FloatBuffer
    val floatBuffer = viewer.getFloatBuffer("float_tensor")

    println("Long Buffer: " + longBuffer)
    println("Float Buffer: " + floatBuffer)
}

def writeTensors(file: File): Unit = {
    val builder = new SafetensorsBuilder()

    // Add a Long array
    val longArray = Array(1L, 2L, 3L, 4L)
    val longShape = Seq(2, 2)
    builder.add("long_tensor", longShape, longArray)

    // Add a FloatBuffer
    val floatBuffer = FloatBuffer.wrap(Array(1.0f, 2.0f, 3.0f, 4.0f))
    val floatShape = Seq(2, 2)
    builder.add("float_tensor", floatShape, floatBuffer)

    // Save to file
    val file = new File("example.safetensors")
    builder.save(file)
}
//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
@main
def main(): Unit = {
  // TIP Press <shortcut actionId="ShowIntentionActions"/> with your caret at the highlighted text
  // to see how IntelliJ IDEA suggests fixing it.
    val path = "D:\\data\\git\\storch-safe-tensors\\example.safetensors"
    val file = new File(path) //"D:\\data\\git\\storch-safe-tensors\\src\\main\\scala\\resources\\model.safetensors")
    val viewer = SafetensorsViewer.load(file)
    println(viewer.header)
}

// TIP To create a run configuration of Scala Application, click <icon src="AllIcons.General.OpenDisk"/> and choose the main class.
//
//    (1 to 5).map(println)
//
//    val file = new File("D:\\data\\git\\storch-safe-tensors\\src\\main\\scala\\resources\\model.safetensors")
//    val viewer = SafetensorsViewer.load(file)
//    println(viewer)
//    for (i <- 1 to 5) do
//      // TIP Press <shortcut actionId="Debug"/> to start debugging your code. We have set one <icon src="AllIcons.Debugger.Db_set_breakpoint"/> breakpoint
//      // for you, but you can always add more by pressing <shortcut actionId="ToggleLineBreakpoint"/>.
//      println(s"i = $i")
