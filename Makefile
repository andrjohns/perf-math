MATH ?=math/
include math/make/compiler_flags
include math/make/libraries

CXXFLAGS+=-O3 -march=native -Ibenchmark/include $(TBB_TARGETS)
LDLIBS+=-lbenchmark
LDFLAGS+=-Lbenchmark/build/src
CXX = clang++

libs: $(TBB_TARGETS)

update: 
	git submodule update --init --recursive

benchmark/build/src/libbenchmark.a: benchmark benchmark/googletest update
	mkdir -p benchmark/build && cd benchmark/build && cmake .. -DCMAKE_BUILD_TYPE=RELEASE && make

benchmark/googletest:
	cd benchmark && git clone https://github.com/google/googletest

##
# Debug target that allows you to print a variable
##
print-%  : ; @echo $* = $($*)
