package torch.safetensors.instance

import java.util.Locale

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.Map

import lombok.AllArgsConstructor
import lombok.Data
import lombok.NoArgsConstructor

/** SafeTensors Plain Old Java Object
  */
//@Data

//enum DataType:
//  case F64, F32, F16, BF16, I64, I32, I16, I8, BOOL
//

object SafeTensors {
  enum DataType:
    case F64, F32, F16, BF16, I64, I32, I16, I8, BOOL

  case class HeaderElement(
      name: String,
      dataType: SafeTensors.DataType,
      shape: Seq[Long],
      offsets: Seq[Long],
  )

  //    @Data
//  @NoArgsConstructor
//  @AllArgsConstructor
//  case class HeaderElement(name: String, dataType: SafeTensors.DataType, shape: Seq[Long], offsets: Seq[Long])

  //  {
  this()
//      this.name = name
//      this.dataType = dataType
//      this.shape = shape
//      this.offsets = offsets
//    }

}

class SafeTensors {
  var sizeOfHeader = 0L
  var header: ListBuffer[SafeTensors.HeaderElement] = null
  var metadata: mutable.Map[String, String] = null
  var filename: String = null

  var name: String = null
  var dataType: SafeTensors.DataType = null
  var shape: Seq[Long] = null
  var offsets: Seq[Long] = null
}
