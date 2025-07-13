# Storch Safe Tensors

## Overview
Storch Safe Tensors is a Scala 3 project designed for reading and writing Hugging Face's Safetensors format files. It enables direct loading of Hugging Face Transformers models in Safetensors format and seamless interoperability with Python. This project aims to provide a robust and efficient solution for handling Safetensors in the Scala ecosystem.

## Features
- **Read and Write Support**: Read from and write to Safetensors format files.
- **Hugging Face Integration**: Directly load Hugging Face Transformers models in Safetensors format.
- **Python Interoperability**: Compatible with Python, allowing cross - language operations.
- **Type Safety**: Leverage Scala's strong type system for safe and reliable operations.

## Prerequisites
- **Scala 3**: This project is built with Scala 3. Make sure you have Scala 3 installed on your system.
- **SBT**: The project uses SBT (Scala Build Tool) for building and managing dependencies. Install SBT from the [official website](https://www.scala-sbt.org/).

## Installation
1. Clone the repository:
```bash
git clone https://github.com/mullerhai/storch-safe-tensor.git
cd torch-safe-tensors
sbt compile
```

```scala 3
libraryDependencies += "io.github.mullerhai" % "storch-safe-tensor_3" % "0.1.0"
```

```scala 3
import torch.safetensors.SafetensorsBuilder
import java.io.File
import java.nio.LongBuffer
import java.nio.FloatBuffer

object WriteExample extends App {
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
```

```scala 3
import torch.safetensors.SafetensorsViewer
import java.io.File

object ReadExample extends App {
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

```


## Python Interoperability
This project can be used in conjunction with Python. You can write Safetensors files in Scala and read them in Python, or vice versa.

Writing in Scala and Reading in Python
Write a Safetensors file in Scala using the SafetensorsBuilder as shown above.
Read the file in Python using the safetensors library:

python
Apply
from safetensors import safe_open
```python


from safetensors.torch import save_file
import torch
with safe_open("example.safetensors", framework="pt", device="cpu") as f:
    long_tensor = f.get_tensor("long_tensor")
float_tensor = f.get_tensor("float_tensor")
long_tensor = torch.tensor([1, 2, 3, 4]).reshape(2, 2)
float_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(2, 2)

tensors = {
    "long_tensor": long_tensor,
    "float_tensor": float_tensor
}

save_file(tensors, "python_example.safetensors")
```

Read the file in Scala using the SafetensorsViewer as shown above.
## Future Plans
Support for More Data Types: Extend support for additional data types.
Performance Optimization: Improve the performance of reading and writing operations.
More File Formats: Add support for other file formats related to large - scale models.
Contributing
Contributions are welcome! Please feel free to submit issues or pull requests on the GitHub repository.

License
This project is licensed under the Apache 2.0 License.
