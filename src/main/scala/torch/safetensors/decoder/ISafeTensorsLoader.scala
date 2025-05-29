package torch.safetensors.decoder

import torch.safetensors.instance.SafeTensors

/** SafeTensors加载器
  */
trait ISafeTensorsLoader[B] {
  def load(tensorsDefinition: SafeTensors, blocks: B): Unit
}
