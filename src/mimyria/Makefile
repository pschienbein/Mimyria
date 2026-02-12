

SRCDIRS := $(wildcard */)
SRC := $(wildcard *.cpp)
SRC += $(foreach I, $(SRCDIRS), $(wildcard $I/*.cpp))
$(info $(SRC))


OBJS := $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(SRC))
DEPS := $(OBJS:.o=.d)

$(info $(OBJS))

PCH := pch.hpp
PCH_GCH := pch.hpp.gch

# -- non-optional libraries
override LDLIBS += -lfftw3_omp -lfftw3 -ldl

TARGET := $(BUILD_DIR)/mimyria

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CXX) $(LDFLAGS) -o $(TARGET) $(OBJS) $(LDLIBS)


$(BUILD_DIR)/%.o: %.cpp $(PCH_GCH) | $(BUILD_DIR)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -include $(PCH) -c $< -o $@


$(PCH_GCH): $(PCH) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ -c $<


$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)


-include $(DEPS)

clean:
	rm -f $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d $(TARGET)

.PHONY: all clean
