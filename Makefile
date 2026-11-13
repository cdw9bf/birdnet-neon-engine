# libbnneon: one NEON engine library: compiled model packages (.bnpk, any model) and the legacy BirdNET v2.4 (.bn24)
# weights file.
# Native build on the Pi (gcc 14) or the Mac (clang, arm64; development only).
# The dotprod-family kernels, the fp16 registry kernels and the v2.4 fp16 kernels are compiled for the Cortex-A76
# (dotprod + fp16); everything else for baseline ARMv8-A, so the same library loads on a Pi 4 (Cortex-A72) and runs
# its fp32 and int8 paths there; the dispatcher checks the CPU features (HWCAP) before entering A76 code.
UNAME := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  CC ?= clang
  A76 = -march=armv8.2-a+dotprod+fp16
  BASE = -march=armv8.2-a+dotprod+fp16
else
  A76 = -mcpu=cortex-a76
  BASE = -march=armv8-a
endif
COMMON = -O3 -fPIC -Wall -Wextra -Wno-unused-parameter -std=gnu11 -ffp-contract=off $(EXTRA)
LDLIBS = -lm -lpthread
ifeq ($(UNAME),Darwin)
  LDLIBS += -framework Accelerate
endif
# the dotprod + fp16 kernels of the neon-fp16 BirdNET v3.0 packages (from the retired hand-written v3 engine)
DOT_SRC = src/gemm_s8.c src/kern_f16.c src/wino.c src/wino_s0.c
V24_A76 ?= src/v24_fast.c src/gemm_f16.c
V24_BASE = src/v24_model.c src/v24_ref.c src/v24_api.c src/v24_fe.c src/bnneon.c src/v24_f32.c src/gemm_f32.c
# package runtime: loader/executor + fp32 registry kernels (baseline), fp16 registry kernels (A76)
RT_A76 = src/rt_ops_f16.c src/pc_kern_f16.c src/pc_proto_s8.c src/rt_ops_dot.c
RT_BASE = src/rt_pkg.c src/rt_pool.c src/rt_ops_f32.c src/rt_ops_f32x.c src/fe_dbimg.c src/wino_f32.c src/rt_ops_s8n.c src/gemm_s8n.c src/proto_s8n.c src/pc_kern_f32.c src/gemv.c
OUT ?= build
OBJ_A76 = $(patsubst src/%.c,$(OUT)/obj/%.o,$(DOT_SRC) $(V24_A76) $(RT_A76))
OBJ_BASE = $(patsubst src/%.c,$(OUT)/obj/%.o,$(V24_BASE) $(RT_BASE))
OBJ = $(OBJ_A76) $(OBJ_BASE)

all: $(OUT)/libbnneon.so $(OUT)/cli

$(OUT)/obj:
	mkdir -p $(OUT)/obj
# header dependencies come from the compiler (-MMD writes obj/<name>.d next to each object; -MP keeps a deleted
# header from breaking the build). The Makefile is a prerequisite so that an edit of its flags rebuilds (flags given on
# the make command line do not), and so that
# objects built before the .d files existed are rebuilt once.
$(OBJ_A76): $(OUT)/obj/%.o: src/%.c Makefile | $(OUT)/obj
	$(CC) $(COMMON) $(A76) -MMD -MP -c -o $@ $<
$(OBJ_BASE): $(OUT)/obj/%.o: src/%.c Makefile | $(OUT)/obj
	$(CC) $(COMMON) $(BASE) -MMD -MP -c -o $@ $<
-include $(OBJ:.o=.d)
$(OUT)/libbnneon.so: $(OBJ)
	$(CC) -shared -o $@ $(OBJ) $(LDLIBS)
$(OUT)/cli: bench/cli.c $(OBJ)
	$(CC) $(COMMON) $(BASE) -o $@ bench/cli.c $(OBJ) $(LDLIBS)
# development tool: the Perch v2 fp32 oracle (readable reference + fake quantization), not part of the library
$(OUT)/perch_oracle: bench/perch_oracle.c src/perch_model.c src/perch_ref.c src/perch_internal.h | $(OUT)/obj
	$(CC) $(COMMON) $(BASE) -o $@ bench/perch_oracle.c src/perch_model.c src/perch_ref.c $(LDLIBS)
# development tool: the thread pool's stress test
$(OUT)/test_pool: bench/test_pool.c src/rt_pool.c src/rt.h | $(OUT)/obj
	$(CC) $(COMMON) $(BASE) -o $@ bench/test_pool.c src/rt_pool.c -lpthread
clean:
	rm -rf $(OUT)
