name      := pro
workdir   := workspace111


# @mkdir -p build && cd build && cmake .. && make -j16
build:

	@echo "------------------- xmake f -v  --------------------"
	@xmake f -v

	@echo "------------------- xmake f --cpu=true -cv  --------------------"
	@xmake f --cpu=true -cv

	@echo "------------------- xmake f --nv-gpu=true --cuda=${CUDA_HOME} -cv  --------------------"
	@xmake f --nv-gpu=true --cuda=${CUDA_HOME} -cv

	@echo "------------------- xmake build && xmake install  --------------------"
	@xmake build && xmake install

	@echo "------------------- make build over  --------------------"


# @make build && cd $(workdir) && ./$(name)
run:
	@make build
	@echo "\n\n\n ------------------- python add_PROFILE_ITERATIONS.py --cuda --profile  -------------------- \n"
	@cd operatorspy/tests/ && /home/ubuntu/miniconda3/envs/py310torch260/bin/python add_PROFILE_ITERATIONS.py --cuda --profile

	@echo "\n\n\n ------------------- python attention_PROFILE_ITERATIONS.py --cuda --profile  -------------------- \n"
	@cd operatorspy/tests/ && /home/ubuntu/miniconda3/envs/py310torch260/bin/python attention_PROFILE_ITERATIONS.py --cuda --profile

	@echo "\n\n\n ------------------- python swiglu_PROFILE_ITERATIONS.py --cuda --profile  -------------------- \n"
	@cd operatorspy/tests/ && /home/ubuntu/miniconda3/envs/py310torch260/bin/python swiglu_PROFILE_ITERATIONS.py --cuda --profile



# 定义清理指令
clean:
	@xmake clean
	@rm -rf  ./build $(workdir)/$(name)

# 防止符号被当做文件
.PHONY : build run clean
