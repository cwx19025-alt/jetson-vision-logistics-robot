from statemachine import StateMachine,State


class RegisterMeta(type):
    def __new__(meta, name, bases, dct):
        cls=super().__new__(meta, name, bases, dct)
        cls.register_dict={}

        for attr_name,attr in dct.items():
            if hasattr(attr,'_register_name'):
                cls.register_dict[attr._register_name]=attr_name

        return cls
    
    def __call__(cls, *args, **kwargs):
        instance=super().__call__(*args, **kwargs)
        for name,method in cls.register_dict.items():
            bound_method=getattr(instance,method)
            instance.task.register_instance_method(name,bound_method)
        return instance











class TaskRequest():  # 注册
    def __init__(self):
        self.register_dict={}

    def register(self,name):  # 注册任务
        def wrapper(func):
            func._register_name=name
            return func
        return wrapper
    
    def get(self,name):
        return self.register_dict.get(name)  # 获取任务
    
    def register_instance_method(self,name,method):  # 注册实例方法
        self.register_dict[name]=method

class Mymachine(StateMachine):  #状态机
    def __init__(self,task_request:TaskRequest):
        self.task_request=task_request
        Mymachine.state0_m.task_request=self.task_request
        Mymachine.state1_c.task_request=self.task_request
        Mymachine.state2_m.task_request=self.task_request
        super().__init__()
        
            
    # ========== 父状态：转盘（复合状态 + 子状态）==========
   
    class state0_m(State.Compound):
        initial = True  # 默认主状态
        # 子状态（）
        state0_wait = State('等待',initial=True)
        state0_move = State('移动')
        state0_stop = State('停止')
        state0_grasp=State('抓取')
        # 子状态跳转（官方写法）
        to_state0_move=state0_wait.to(state0_move)
        to_state0_wait=state0_move.to(state0_wait)
        to_state0_stop=state0_move.to(state0_stop) | state0_grasp.to(state0_stop)
        to_state0_grasp=state0_stop.to(state0_grasp)

        def on_enter_state0_move(self):   # 进入状态0_move，移动
            print("进入状态0_move，移动")
            print("0000000000000000")
            self.task_request.get("to0_move")()
            
        def on_enter_state0_wait(self):
            print("进入状态0_wait，等待")
            print("1111111111111111")
            self.task_request.get("to0_wait")()

        def on_enter_state0_stop(self):
            print("进入状态0_stop，停止")
            print("2222222222222222")
            # self.task_request.get("to0_stop")()

        def on_enter_state0_grasp(self):
            print("进入状态0_grasp，抓取")
            print("3333333333333333")   
            # self.task_request.get("to0_grasp")()

    class state1_c(State.Compound):
        
        # 子状态（）
        state1_move = State('移动',initial=True)
        state1_grasp=State('抓取')
        # 子状态跳转（官方写法）
        to_state1_grasp=state1_grasp.to(state1_move)
        to_state1_move=state1_move.to(state1_grasp)


    class state2_m(State.Compound):
        # 子状态（）
        state2_move = State('移动',initial=True)
        state2_grasp=State('抓取')
        # 子状态跳转（官方写法）
        to_state2_grasp=state2_grasp.to(state2_move)
        to_state2_move=state2_move.to(state2_grasp)

    class state3_c(State.Compound):
        
        
        state3_move = State('移动',initial=True)
        state3_grasp=State('抓取')

        to_state3_grasp=state3_grasp.to(state3_move)
        to_state3_move=state3_move.to(state3_grasp)


    class state4_m(State.Compound):
       
        
        state4_move=State('移动',initial=True)
        state4_grasp=State('抓取')

        to_state4_grasp=state4_grasp.to(state4_move)
        to_state4_move=state4_move.to(state4_grasp)

    to_state0=(state1_c.to(state0_m) | state2_m.to(state0_m)|state3_c.to(state0_m)|state4_m.to(state0_m))
    to_state1=(state0_m.to(state1_c) | state2_m.to(state1_c) | state3_c.to(state1_c) | state4_m.to(state1_c))
    to_state2=(state0_m.to(state2_m) | state1_c.to(state2_m) | state3_c.to(state2_m) | state4_m.to(state2_m))
    to_state3=(state2_m.to(state3_c) | state4_m.to(state3_c) | state0_m.to(state3_c) | state1_c.to(state3_c))
    to_state4=(state3_c.to(state4_m) | state0_m.to(state4_m) | state1_c.to(state4_m) | state2_m.to(state4_m))



    

    def on_exit_state0_m(self,*args, **kwargs):   # 进入状态0，目标转盘
        print("离开状态0，转盘")
        circle=kwargs.get("circle")
        self.task_request.get("to0")(circle)
    
    def on_enter_state1_c(self,*args, **kwargs):   # 进入状态1，目标绿圈
        print("进入状态1，绿圈")
        circle=kwargs.get("circle")
        self.task_request.get("to1")(circle)



    def on_enter_state2_m(self,*args, **kwargs):   # 进入状态2，目标绿材料
        print("进入状态2，绿材料")
        circle=kwargs.get("circle")
        self.task_request.get("to2")(circle)
    
    def on_enter_state3_c(self,*args, **kwargs):   # 进入状态3，目标绿圈
        print("进入状态3，绿圈")
        circle=kwargs.get("circle")
        self.task_request.get("to3")(circle)

  
    def on_enter_state4_m(self,*args, **kwargs):   # 进入状态4，目标绿材料
        print("进入状态4，绿材料")
        circle=kwargs.get("circle")
        self.task_request.get("to4")(circle)
        

class test(metaclass=RegisterMeta):
    task = TaskRequest()
    def __init__(self) -> None:
        pass
    @task.register("to0_wait")
    def to0_wait(self):
        print(f"✅ 执行 to0 等待动作")
    
    @task.register("to0_move")
    def to0_move(self):
        print(f"✅ 执行 to0 转111111111111111")
    
    @task.register("to0")   
    def to0(self,circle=None):
        print(f"✅ 执行 to0 转盘动作 | circle={circle}")

    @task.register("to1")   
    def to1(self,circle=None):
        print(f"✅ 执行 to1 绿圈动作 | circle={circle}")

    @task.register("to2")       
    def to2(self,circle=None):
        print(f"✅ 执行 to2 绿材料动作 | circle={circle}")

    @task.register("to3")           
    def to3(self,circle=None):
        print(f"✅ 执行 to3 绿圈动作 | circle={circle}")

    @task.register("to4")
    def to4(self,circle=None):
        print(f"✅ 执行 to4 绿材料动作 | circle={circle}")

    def main(self):
        # 3. 创建状态机
        sm = Mymachine(self.task)

        print("===== 开始测试状态切换 =====")

        # 切换主状态（带参数）
        sm.send("to_state0_move")
        print(len(sm.configuration_values))
        
        if sm.configuration_values[1]=='state0_move':
            print(sm.configuration_values[1])
        else:
            print("其他状态")
            print(sm.configuration_values[1])




        sm.send("to_state0_wait")

        sm.send("to_state1", circle=10)    # 0 → 1
        
        sm.send("to_state2", circle=20)    # 1 → 2
        sm.send("to_state3", circle=30)    # 2 → 3
        print(sm.configuration)
        sm.send("to_state4", circle=40)    # 3 → 4
        sm.send("to_state0", circle=0)     # 4 → 0



    


if __name__ == '__main__':

    T=test()
    T.main()