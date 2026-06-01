from io import BufferedWriter, BufferedReader, SEEK_SET, SEEK_END, SEEK_CUR
from typing import Iterable, overload, BinaryIO
import numpy as np
import struct
from enum import IntEnum, IntFlag
from dataclasses import dataclass
from pathlib import Path
import math

from ..rl.base_nn_optimiser import NetworkOptimiser
from ..rl.openaies import OpenAIES
from ..rl.ppoptimiser import PPOptimiser
from ..rl.sac_optimiser import SACOptimiser

from .parsing_functions import (
    read_nnmodule,
    read_normalizer,
    read_optimizer,
    read_hyperparams,
    write_nnmodule,
    write_normalizer,
    write_optimizer,
    write_hyperparams,

    read_sac_rollout,
    write_sac_rollout,
    )

class Member[_]:
    """Member of the given class"""

# --- Format Constants -----------------------------------------------

# file format name so we know we're not loading a png or something
MAGIC = b"SIMTORCH"
VERSION = 1

class SectionID(IntEnum):
    HEADER = 0 # always first, so ommitted
    METADATA = 1
    NETWORK = 2
    TENSOR = 3

class FileFlags(IntFlag):
    NONE = 0
    INITIALIZED = 1 << 0
    HAS_OPTIMIZER = 1 << 1

# --- Section Directory ----------------------------------------------
@dataclass(slots=True)
class SectionEntry:
    """Data about a section's position in the file"""
    id: Member[SectionID]
    offset: int
    length: int

@dataclass(slots=True)
class TensorEntry:
    """Info about a tensor. Offset is relative to header start, and allows you to read the numpy array at that offset"""
    #! I have no clue if this will work. Calculating offsets might be hell
    tensor_id: int
    offset: int


# --- Base Stream ----------------------------------------------------
class FileInterface:
    """Shared low-level binary helpers.
    Reader and writer both inherit this
    
    Format:
    [HEADER](no preamble)
    magic:byte[8] -> SIMTORCH
    version:u32
    flags:u32 (initialized, has_optimizer)
    step:u64 (epoch step)
    env_name_len:u8
    env_name:byte[env_name_len]
    algo_name_len:u8
    algo_name:byte[algo_name_len]

    [METADATA](preamble:u8, length:u64)
    metadata:byte[length]

    [NETWORK](preamble:u8, length:u64)
    (next depends on algorithm)

    [TENSOR](preamble:u8, length:u64)
    count:u32
    lengths:[array_length:u64][count]
    arrays:[data:Any][array_length][count]
    
    """
    def __init__(self, file:BufferedWriter|BufferedReader|BinaryIO):
        self.file = file

        self.version:int = VERSION

        self.flags:FileFlags = FileFlags.NONE
        self.step: int = 0
        self.metadata:str = ""
        self.__env:str = ""
        self.__algorithm:str = ""

        self.sections: dict[SectionID, SectionEntry] = {}
        self.tensor_table: dict[int, TensorEntry] = {}

    @property
    def env(self) -> str:
        return self.__env
    @env.setter
    def env(self, value:str):
        if len(value.encode("utf-8")) > 255:
            raise Exception("Environment name length too large to save")
        self.__env = value

    @property
    def algorithm(self) -> str:
        return self.__algorithm
    @algorithm.setter
    def algorithm(self, value:str):
        if len(value.encode("utf-8")) > 255:
            raise Exception("algorithm name length too large to save")
        self.__algorithm = value


    def tell(self) -> int:
        """
        Return current stream position
        """
        return self.file.tell()
    
    def seek(self, offset: int, whence: int = SEEK_SET):
        """
        Change the stream position to the given byte offset.
            offset
                The stream position, relative to 'whence'.

            whence
                The relative position to seek from.

        The offset is interpreted relative to the position indicated by whence. Values for whence are:

        - os.SEEK_SET or 0 -- start of stream (the default); offset should be zero or positive
        - os.SEEK_CUR or 1 -- current stream position; offset may be negative
        - os.SEEK_END or 2 -- end of stream; offset is usually negative

        Return the new absolute position.
        """
        # docstring taken from this definition
        return self.file.seek(offset, whence)
    
    def align_fill(self, boundary: int = 8):
        """Fills the file with 0s until the next <boundary> byte address"""
        pos = self.tell()
        padding = (-pos) % boundary
        if padding:
            self.file.write(b'\x00' * padding)

    def align_move(self, boundary:int):
        """Moves the ptr to align it to th next <boundary> byte address"""
        pos = self.tell()
        padding = (-pos) % boundary
        if padding:
            self.seek(pos + padding)

    #& primitive writes
    def write_u8(self, value: int):
        """Writes a unsigned 8-bit integer (1 byte)"""
        self.file.write(struct.pack("<B", value))

    def write_u16(self, value:int):
        """Writes a unsigned 16-bit integer (2 bytes)"""
        self.file.write(struct.pack("<H", value))

    def write_u32(self, value: int):
        """Writes a unsigned 32-bit integer (4 bytes)"""
        self.file.write(struct.pack("<I", value))

    def write_u64(self, value: int):
        """Writes a unsigned 64-bit integer (8 bytes)"""
        self.file.write(struct.pack("<Q", value))

    def write_i16(self, value: int):
        """Writes a signed 16-bit integer (2 bytes)"""
        self.file.write(struct.pack("<h", value))

    def write_i32(self, value: int):
        """Writes a signed 32-bit integer (4 bytes)"""
        self.file.write(struct.pack("<i", value))

    def write_i64(self, value: int):
        """Writes a signed 64-bit integer (8 bytes)"""
        self.file.write(struct.pack("<q", value))

    def write_f32(self, value: float | None):
        """Writes a 32-bit floating point number (float) (4 bytes). Around 7-8 decimals of precision around zero
        
        Can also encode a None value. To do this, writes a (positive) NaN float and adds a payload of 255 to the mantissa
        """
        if value is None:
            our_nan = (255 << 23) | 255
            self.write_i32(our_nan)
        else:
            self.file.write(struct.pack("<f", value))

    def write_f64(self, value: float | None):
        """Writes a 64-bit floating point number (double) (8 bytes). Around 16 decimals of precision around zero
        
        Can also encode a None value. To do this, writes a (positive) NaN double and adds a payload of 255 to the mantissa
        """
        if value is None:
            our_nan = (2047 << 52) | 255
            self.write_i64(our_nan)
        else:
            self.file.write(struct.pack("<d", value))

    def write_bool(self, value: bool):
        """Writes a boolean value (1 byte)"""
        self.file.write(struct.pack("<?", value))

    def write_str(self, value: str):
        """writes up to 255 characters as bytes in utf-8. Uses the first byte to describe the length
        
        Equivalent to a pascal string (or so I've been told)"""
        data = value.encode("utf-8")
        if len(data) > 255: raise ValueError("string too long to be encoded")
        self.write_u8(len(data))
        self.file.write(data)

    #& primitive reads
    def read_u8(self) -> int:
        """Reads a unsigned 8-bit integer (1 byte)"""
        return struct.unpack("<B", self.file.read(1))[0]
    
    def read_u16(self) -> int:
        """Reads a unsigned 16-bit integer (2 bytes)"""
        return struct.unpack("<H", self.file.read(2))[0]

    def read_u32(self) -> int:
        """Reads a unsigned 32-bit integer (4 bytes)"""
        return struct.unpack("<I", self.file.read(4))[0]

    def read_u64(self) -> int:
        """Reads a unsigned 64-bit integer (8 bytes)"""
        return struct.unpack("<Q", self.file.read(8))[0]
    
    def read_i16(self) -> int:
        """Reads a signed 16-bit integer (2 bytes)"""
        return struct.unpack("<h", self.file.read(2))
    
    def read_i32(self) -> int:
        """Reads a signed 32-bit integer (4 bytes)"""
        return struct.unpack("<i", self.file.read(4))[0]

    def read_i64(self) -> int:
        """Reads a signed 64-bit integer (8 bytes)"""
        return struct.unpack("<q", self.file.read(8))[0]

    def read_f32(self) -> float:
        """Reads a 32-bit floating point number (float) (4 bytes). Around 7-8 decimals of precision around zero
        
        Can also decode a None value. To do this, checks whether the payload of the (positive) NaN is 255
        """
        data = struct.unpack("<f", self.file.read(8))[0]
        if math.isnan(data):
            our_nan = (255 << 23) | 255
            if struct.unpack("<i", struct.pack("<f", data)) == our_nan:
                return None
        return data
    
    def read_f64(self) -> float:
        """Reads a 64-bit floating point number (float) (8 bytes). Around 16 decimals of precision around zero
        
        Can also decode a None value. To do this, checks whether the payload of the (positive) NaN is 255
        """
        data = struct.unpack("<d", self.file.read(8))[0]
        if math.isnan(data):
            our_nan = (2047 << 52) | 255
            if struct.unpack("<q", struct.pack("<d", data)) == our_nan:
                return None
        return data

    def read_bool(self) -> bool:
        """Reads a boolean value (1 byte)"""
        return struct.unpack("<?", self.file.read(1))[0]

    def read_str(self) -> str:
        """reads up to 255 utf-8 bytes as a str
        
        Equivalent to a pascal string"""
        length = self.read_u8()
        return self.file.read(length).decode("utf-8")
    
# --- Writer ---------------------------------------------------------

class Writer(FileInterface):

    SECTION_PREAMBLE = 1+8 # u8 section ID + u64 offset

    def __init__(self, path:str | Path):
        self.path = Path(path)
        super().__init__(open(self.path, "wb"))

        self._section_order: list[SectionEntry] = []

        self._tensor_ids: dict[int, int] = {}
        self._tensors: list[np.ndarray] = []

    @property
    def initialized(self) -> bool:
        return bool(self.flags & FileFlags.INITIALIZED)

    @property
    def has_optimizer(self) -> bool:
        return bool(self.flags & FileFlags.HAS_OPTIMIZER)
    
    def reserve(self, amount:int):
        """Pad the file with the amount of bytes given"""
        self.file.write(b'\x00' * amount)

    def register_tensor(self, tensor:np.ndarray):
        """Registers the tensor (will be written to the file later). Writes its static lookup id to the current file position"""
        ptr = id(tensor)
        # get index of array from [ptr : id] map
        exists = self._tensor_ids.get(ptr)
        if exists is not None:
            self.write_u32(exists)
            return
        
        tensor_id = len(self._tensors)
        self._tensor_ids[ptr] = tensor_id
        self._tensors.append(tensor)
        self.write_u32(tensor_id)


    def write_header_section(self):
        """
        File Layout:
        [magic:8]
        [version:u32]
        [flags:u32]
        [step:u64]
        [env:str]
        [algorithm:str]
        """

        self.file.write(MAGIC)

        self.write_u32(self.version)
        self.write_u32(int(self.flags))
        self.write_u64(self.step)
        self.write_str(self.env)
        self.write_str(self.algorithm)
        self._section_order.append(SectionEntry(SectionID.HEADER, 0, self.tell()))

    def write_tensor_section(self):
        """
        File Layout:
        [LookupTableLength:u32]
        [size:u64][]
        [Tensor:numpy array][]
        """
        # We should encode the array lengths, not the offset. Else, it's not clear what the offset is relative to
        # Is it relative to the lookup table start? The section start? The lookup table entry?
        self.begin_section(SectionID.TENSOR)
        tensor_count = len(self._tensors)
        self.write_u32(tensor_count)
        lookup_table_start = self.tell()

        # 1. Write the lookup table (reserve space for each size def.)
        # 2. Write each numpy array after this
        # 3. Go back to lookuptable and write the sizes

        for i in range(tensor_count):
            # reserve enough for a u64 offset
            self.reserve(8)

        for i, tensor in enumerate(self._tensors):
            start = self.tell()
            #? pickle=False should allow for better compatibility long-term
            np.save(self.file, tensor, allow_pickle=False)
            end = self.tell()
            tensor_size = end - start

            # go back to the start of the lookup table, where we reserved space to write these sizes
            self.seek(lookup_table_start + i*8)

            self.write_u64(tensor_size)

            self.seek(end)

        # Then, to get start of tensor array, add the cumulative sum of all the array lengths from 0 to n, then add that to the end of the lookup table
        # remember that tensors[0] starts at the end of the lookup table

        self.end_section(SectionID.TENSOR)

    def write_metadata_section(self):
        self.begin_section(SectionID.METADATA)
        self.file.write(bytes(self.metadata.encode("utf-8")))
        self.end_section(SectionID.METADATA)
    
#//    def write_network_section(self, network:NetworkOptimiser):
#//        self.begin_section(SectionID.NETWORK)
#//
#//        if type(network) == OpenAIES:
#//            # 1. write hyperparams
#//            #& write_hyperparams()
#//            # 2. write observation normaliser
#//            write_normalizer(network.obs_normalizer, self)
#//            # 3. write main network
#//            write_nnmodule(network.main_network, self)
#//            # 4. write optimizer
#//            write_optimizer(network.optimizer, self)
#//
#//        elif type(network) == PPOptimiser:
#//            # 1. write hyperparams
#//            #& write_hyperparams()
#//            # 2. write observation normaliser
#//            write_normalizer(network.obs_normalizer, self)
#//            # 3. write actor module and optimizer
#//            write_nnmodule(network.actor, self)
#//            write_optimizer(network.actor_optimizer, self)
#//            # 4. write critic module and optimizer
#//            write_nnmodule(network.critic, self)
#//            write_optimizer(network.critic_optimizer, self)
#//
#//        elif type(network) == SACOptimiser:
#//            # 1. write hyperparams
#//            #& write_hyperparams(network.hyperparams, self)
#//            # 2. write observation normaliser
#//            write_normalizer(network.obs_normalizer, self)
#//            # 3. write the pi (actor) network
#//            write_nnmodule(network.pi, self)
#//            write_optimizer(network.pi_optimizer, self)
#//            # 4. Write the qs and q_tarrs twins
#//            write_nnmodule(network.qs[0], self)
#//            write_nnmodule(network.q_tarrs[0], self)
#//            write_optimizer(network.q_optimizers[0], self)
#//
#//            write_nnmodule(network.qs[1], self)
#//            write_nnmodule(network.q_tarrs[1], self)
#//            write_optimizer(network.q_optimizers[1], self)
#//        
#//        else:
#//            raise TypeError(f"Unknown network optimiser {network}")
#//
#//        self.end_section(SectionID.NETWORK)




    def begin_section(self, section_id: Member[SectionID]) -> int:
        """Begins the section. Returns the start of usable data"""
        #? Should section start include or exclude the preamble?
        # since we align to 16 bytes before inserting a section, but the previous section doesn't include that padding,
        # in its section width, we need to align when reading sections too so the offset takes the padding into account
        self.align_fill(16)
        # start section entry. We'll fill the length and go back to write it here later
        self.write_u8(int(section_id))
        self.reserve(8)
        start = self.tell()
        self._section_order.append(SectionEntry(section_id, start, None))
        return self.tell()
    
    def end_section(self, section_id: Member[SectionID]):
        """Finalize section. Calculates size and writes to section header"""
        end = self.tell()
        for entry in self._section_order:
            if entry.id == section_id and entry.length == None:
                # move to start of section, write length, then go back to end
                entry.length = (end - entry.offset)
                # the offset gives start of data. We want to write before that, to the 8 bytes we reserved earlier
                self.seek(entry.offset-8)
                self.write_u64(entry.length)
                self.seek(end)
                return
        raise ValueError("Could not find the start of this section")



# --- Reader ---------------------------------------------------------
class Reader(FileInterface):
    def __init__(self, path:str|Path):
        self.path = Path(path)
        super().__init__(open(self.path, "rb"))

    @property
    def initialized(self) -> bool:
        return bool(self.flags & FileFlags.INITIALIZED)

    @property
    def has_optimizer(self) -> bool:
        return bool(self.flags & FileFlags.HAS_OPTIMIZER)

    def parse_header(self):
        """Reads the file extension, version, epoch step, environment id, and network optimiser algorithm"""
        #todo check how to get byte count of bytes text
        magic = self.file.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError("Invalid file format")
        
        self.version = self.read_u32()
        self.flags = FileFlags(self.read_u32())
        self.step = self.read_u64()
        self.env = self.read_str()
        self.algorithm = self.read_str()

        # also jump ahead and read all the sections
        self.sections[SectionID.HEADER] = SectionEntry(SectionID.HEADER, 0, self.tell())
        while self.file.peek() != b'':
            # move forward so we match the padding done when writing the section
            self.align_move(16)
            section_id = self.read_u8()
            enum = SectionID(section_id)
            length = self.read_u64()
            # we don't include the preamble in our reads because we move while parsing it
            # that means goto_section will use the location of data start which is good
            end = self.tell() + length
            self.sections[enum] = SectionEntry(enum, self.tell(), length)
            self.seek(end)
        #todo [change] go to metadata section preamble (?)
        self.seek(self.sections[SectionID.HEADER].length)

    def parse_metadata(self):
        """Decodes the File interface's string metadata as utf-8 data in the metadata section"""
        self.goto_section(SectionID.METADATA)
        self.metadata = self.file.read(self.sections[SectionID.METADATA].length).decode("utf-8")

    # do this before parsing network else it'll lookup error on read_tensor()
    def parse_tensor(self):
        self.goto_section(SectionID.TENSOR)

        self.arrays:list[np.ndarray] = []
        count = self.read_u32()
        lookup_table_start = self.tell()
        # print("lookup_table_start",lookup_table_start)
        # start of the arrays
        array_start = lookup_table_start + count * 8
        offset_total = 0
        for i in range(count):
            self.seek(array_start + offset_total)
            self.arrays.append(np.load(self.file, allow_pickle=False))
            self.seek(lookup_table_start + i * 8)
            width = self.read_u64()
            # print("incrementing array lookup location by", width)
            offset_total += width


    #! TODO store the network optimizer in the file header
    #! TODO store enough to init the environment in the file header too (or metadata)
#//    def parse_network_section(self, env, optimizer:NetworkOptimiser) -> NetworkOptimiser:
#//        self.goto_section(SectionID.NETWORK)
#//
#//        if optimizer == OpenAIES:
#//            hp = read_hyperparams(self)
#//            norm = read_normalizer(self)
#//            network = read_nnmodule(self)
#//            optim = read_optimizer(self)
#//
#//            cls = OpenAIES(env, network, optim)
#//            cls._OpenAIES__obs_normalizer = norm
#//            cls._OpenAIES__hyper_params = hp
#//            return cls
#//        
#//        if optimizer == PPOptimiser:
#//            hp = read_hyperparams(self)
#//            norm = read_normalizer(self)
#//            actor = read_nnmodule(self)
#//            actor_optim = read_optimizer(self)
#//
#//            critic = read_nnmodule(self)
#//            critic_optim = read_optimizer(self)
#//
#//            cls = PPOptimiser(env, [actor, critic], actor_optim, critic_optim)
#//            cls._PPOptimiser__obs_normalizer = norm
#//            cls._PPOptimiser__hyper_params = hp
#//            return cls
#//        
#//        if optimizer == SACOptimiser:
#//            hp = read_hyperparams(self)
#//            norm = read_normalizer(self)
#//
#//            actor = read_nnmodule(self)
#//            actor_optim = read_optimizer(self)
#//
#//            qs = [None,None]
#//            q_tarrs = [None, None]
#//            q_optim = [None, None]
#//            qs[0] = read_nnmodule(self)
#//            q_tarrs[0] = read_nnmodule(self)
#//            q_optim[0] = read_optimizer(self)
#//
#//            qs[1] = read_nnmodule(self)
#//            q_tarrs[1] = read_nnmodule(self)
#//            q_optim[1] = read_optimizer(self)
#//
#//            cls = SACOptimiser(env, actor, qs, actor_optim, q_optim)
#//            cls._SACOptimiser__q_tarrs = q_tarrs
#//            cls._SACOptimiser__obs_normalizer = norm
#//            cls._SACOptimiser__hyper_params = hp
#//            return cls
#//        raise Exception(f"Unknown optimiser type {optimizer}")

            


    def goto_section(self, section:Member[SectionID]):
        entry = self.sections.get(section)
        if entry is None:
            raise KeyError(f"Missing section {section}")
        
        self.seek(entry.offset)


    def read_tensor(self) -> np.ndarray:
        """Reads a u32 from the file and uses that as the tensor id to lookup"""
        tensor_id = self.read_u32()
        return self.arrays[tensor_id]
    


if __name__ == "__main__":
    from ..core.nn import Tanh
    from .parsing_functions import write_Tanh, MODULE_MAP
    my_tanh = Tanh()

    writer = Writer("./test.st")
    print(MODULE_MAP)
    write_nnmodule(my_tanh, writer)
    writer.file.close()
    
