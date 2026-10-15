import serial
import time
import glob






class uart_protocol:
    HEADER_LEN=2
    CMD_LEN=1
    
    LENGTH_LEN=1
    CHECKSUM_LEN=1
    def __init__(self,port,baudrate=115200):
        self.ser = None
        self.port = port
        self.baudrate = baudrate
        self.header_flag=False
        self.cmd_id_flag=False
        self.length_flag=False
        self.content_flag=False
        self.checksum_flag=False

        self.byte_buffer={}

        self.header_receive=b''
        self.cmd_id_receive=b'' 
        self.length_receive=b''
        self.content_receive=b''
        self.checksum_receive=b''  

        self.checksum_fail=False
        self.connect()

    def connect(self):
        while self.ser is None:
            print(f"尝试连接串口 {self.port}")
            time.sleep(1)
            try:
                self.ser = serial.Serial(self.port, self.baudrate)
                print("串口连接成功")
            except (serial.SerialException, FileNotFoundError):
                print(f"无法打开串口 {self.port}")
                acm_ports = glob.glob('/dev/ttyACM*')
                if acm_ports:
                    self.port = acm_ports[0]
                else:
                    print("未找到可用的 ACM 端口")
                    self.ser = None
                    time.sleep(5)


            except KeyboardInterrupt:
                print("用户中断连接尝试")
                self.ser = None
                exit()

    def send_test(self):
        header=b'\x05\x1c'
        cmd_id=b'\x02'
        length=b'\x01'
        content=b'\x01'
        checksum=(sum(header+cmd_id+length+ content)&0xFF).to_bytes(1,'little')
        packet=header+cmd_id+length+content+checksum
        print(packet)
        # try:
        #     self.ser.write(packet)
        # except serial.SerialException:
        #     print("写入数据失败")
        #     self.ser = None
        #     time.sleep(3)
        #     self.connect()

    def send_data_rgb(self,data):
        header=b'\x05\x1c'
        cmd_id=b'\x01'
        length=b'\x07'
        content=data
        checksum=(sum(header+cmd_id+length+ content)&0xFF).to_bytes(1,'little')
        packet=header+cmd_id+length+content+checksum
        try:
            self.ser.write(packet)
        except serial.SerialException:
            print("写入数据失败")
            self.ser = None
            time.sleep(3)
            self.connect()

    def send_start(self):
        header=b'\x12\x4c'
        cmd_id=b'\x00'
        length=b'\x00'
        content=b'\x01'
        checksum=(sum(header+cmd_id+length+ content)&0xFF).to_bytes(1,'little')
        packet=header+cmd_id+length+content+checksum
        try:
            self.ser.write(packet)
        except serial.SerialException:
            print("写入数据失败")
            self.ser = None
            time.sleep(3)
            self.connect()

    def receive_data(self):
        try:
            if self.ser.in_waiting > 0:
                data = self.ser.read(self.ser.in_waiting)
                print(f"收到数据: {data}")
        except serial.SerialException:
            print("读取数据失败")
            self.ser = None
            time.sleep(3)
            self.connect()

    def buffer_clear(self):
        self.content_length=0
        self.header_receive=b''
        self.cmd_id_receive=b'' 
        self.length_receive=b''
        self.content_receive=b''
        self.checksum_receive=b''  

        self.header_flag=False
        self.cmd_id_flag=False
        self.length_flag=False
        self.content_flag=False
        self.checksum_flag=False
    def receive_by_byte(self,new_byte):

            

        if not self.header_flag:
            if len(self.header_receive)==self.HEADER_LEN-1:
                self.header_receive+=new_byte
                if self.header_receive==b'\x12\x4c':
                    self.header_flag=True

                else:
                    self.header_receive=b''
                    self.header_flag=False
                    return
                    
            else:
                if new_byte==b'\x12':
                    self.header_receive+=new_byte
                else:

                    return
        
        elif not self.cmd_id_flag:
            self.cmd_id_flag=True
            self.cmd_id_receive=new_byte

        elif not self.length_flag:
            self.length_flag=True
            self.length_receive+=new_byte
            self.content_length=int.from_bytes(self.length_receive,'little')
        elif not self.content_flag:
            
            self.content_receive+=new_byte
            if len(self.content_receive)==self.content_length:
                self.content_flag=True
            
        elif not self.checksum_flag:
            self.checksum_flag=True
            self.checksum_receive+=new_byte
            if self.checksum_receive==(sum(self.header_receive+self.cmd_id_receive+self.length_receive+self.content_receive)&0xFF).to_bytes(self.CHECKSUM_LEN,'little'):
                print('校验成功')
                self.byte_buffer={
                    'header':self.header_receive,
                    'cmd_id':self.cmd_id_receive,
                    'length':self.length_receive,
                    'content':self.content_receive,
                    'checksum':self.checksum_receive,
                }
                print(self.byte_buffer)
                
                self.buffer_clear()
                    

            else:
                print('校验失败')
                self.buffer_clear()
                self.checksum_fail=True

    #接收数据
    def rec(self):
        while True:
            if self.ser.in_waiting>0:

                new_byte=self.ser.read(1)
                self.receive_by_byte(new_byte)

            else:
                time.sleep(0.01)



if __name__ == "__main__":
    uart = uart_protocol("com12",115200)
    uart.rec()


