/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CORE_MINDRT_INCLUDE_ACTOR_BUSLOG_H
#define MINDSPORE_CORE_MINDRT_INCLUDE_ACTOR_BUSLOG_H

#include <signal.h>
#include <iostream>

#include <sstream>
#include <string>

#include "actor/buserrcode.h"
#ifdef USE_GLOG
#include "utils/log_adapter.h"
#else
#include "common/log_adapter.h"
#endif
namespace mindspore {
#define BUS_LOG(severity)       // LOG(severity)
#define BUS_DLOG(verboselevel)  // VLOG(verboselevel)

#define HARES_LOG_PID int  // GetLogPID();
#define PID_LITEBUS_LOG

#define ICTSBASE_LOG_COMMON_CODE
#define HLOG_LEVEL_INFO
#define PID_LITEBUS_LOG
#define HLOG_LEVEL_DEBUG 1
#define ICTSBASE_LOG0(logig, level, pid, format)
#define ICTSBASE_LOG1(logig, level, pid, format, para)
#define ICTSBASE_LOG2(logig, level, pid, format, para1, para2)
#define ICTSBASE_LOG3(logig, level, pid, format, para1, para2, para3)
#define ICTSBASE_LOG4(logig, level, pid, format, para1, para2, para3, para4)
#define ICTSBASE_LOG_STRING(logig, level, pid, preformat, format...)
#define FlushHLogCache()
// Kill the process for safe exiting.
inline void KillProcess(const std::string &ret) {
  ICTSBASE_LOG_STRING(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_INFO, PID_LITEBUS_LOG, "BUS Exit Tip: %s", "%s",
                      ret.c_str());
  // flush the log in cache to disk before exiting.
  FlushHLogCache();
}
}  // namespace mindspore
constexpr int DLEVEL4 = 1000;
constexpr int DLEVEL3 = 3;
constexpr int DLEVEL2 = 2;
constexpr int DLEVEL1 = 1;
constexpr int DLEVEL0 = 0;

#define BUS_ASSERT(expression)                                                                       \
  do {                                                                                               \
    if (!(expression)) {                                                                             \
      std::stringstream ss;                                                                          \
      ss << "Assertion failed: " << #expression << ", file: " << __FILE__ << ", line: " << __LINE__; \
      mindspore::KillProcess(ss.str());                                                              \
    }                                                                                                \
  } while (0)

#define BUS_EXIT(ret)                                                           \
  do {                                                                          \
    std::stringstream ss;                                                       \
    ss << (ret) << "  ( file: " << __FILE__ << ", line: " << __LINE__ << " )."; \
    mindspore::KillProcess(ss.str());                                           \
  } while (0)

#define BUS_OOM_EXIT(ptr)                                                                                  \
  {                                                                                                        \
    if (ptr == nullptr) {                                                                                  \
      ICTSBASE_LOG0(ICTSBASE_LOG_COMMON_CODE, HLOG_LEVEL_ERROR, PID_LITEBUS_LOG, "new failed, will exit"); \
      BUS_EXIT("Exit for OOM.");                                                                           \
    }                                                                                                      \
  }

constexpr int LOG_CHECK_EVERY_FIRSTNUM = 10;
constexpr int LOG_CHECK_EVERY_NUM1 = 10;
constexpr int LOG_CHECK_EVERY_NUM2 = 100;
constexpr int LOG_CHECK_EVERY_NUM3 = 1000;
constexpr int LOG_CHECK_EVERY_NUM4 = 10000;

#define LOG_CHECK_ID_CONCAT(word1, word2) word1##word2

#define LOG_CHECK_ID LOG_CHECK_ID_CONCAT(__FUNCTION__, __LINE__)

#define LOG_CHECK_FIRST_N              \
  [](uint32_t firstNum) {              \
    static uint32_t LOG_CHECK_ID = 0;  \
    ++LOG_CHECK_ID;                    \
    return (LOG_CHECK_ID <= firstNum); \
  }

#define LOG_CHECK_EVERY_N1                                            \
  [](uint32_t firstNum, uint32_t num) {                               \
    static uint32_t LOG_CHECK_ID = 0;                                 \
    ++LOG_CHECK_ID;                                                   \
    return ((LOG_CHECK_ID <= firstNum) || (LOG_CHECK_ID % num == 0)); \
  }

#define LOG_CHECK_EVERY_N2                                                                     \
  [](uint32_t firstNum, uint32_t num1, uint32_t num2) {                                        \
    static uint32_t LOG_CHECK_ID = 0;                                                          \
    ++LOG_CHECK_ID;                                                                            \
    return ((LOG_CHECK_ID <= firstNum) || (LOG_CHECK_ID < num2 && LOG_CHECK_ID % num1 == 0) || \
            (LOG_CHECK_ID % num2 == 0));                                                       \
  }

#define LOG_CHECK_EVERY_N3                                                                     \
  [](uint32_t firstNum, uint32_t num1, uint32_t num2, uint32_t num3) {                         \
    static uint32_t LOG_CHECK_ID = 0;                                                          \
    ++LOG_CHECK_ID;                                                                            \
    return ((LOG_CHECK_ID <= firstNum) || (LOG_CHECK_ID < num2 && LOG_CHECK_ID % num1 == 0) || \
            (LOG_CHECK_ID < num3 && LOG_CHECK_ID % num2 == 0) || (LOG_CHECK_ID % num3 == 0));  \
  }

#define LOG_CHECK_EVERY_N4                                                                                            \
  [](uint32_t firstNum, uint32_t num1, uint32_t num2, uint32_t num3, uint32_t num4) {                                 \
    static uint32_t LOG_CHECK_ID = 0;                                                                                 \
    ++LOG_CHECK_ID;                                                                                                   \
    return ((LOG_CHECK_ID <= firstNum) || (LOG_CHECK_ID < num2 && LOG_CHECK_ID % num1 == 0) ||                        \
            (LOG_CHECK_ID < num3 && LOG_CHECK_ID % num2 == 0) || (LOG_CHECK_ID < num4 && LOG_CHECK_ID % num3 == 0) || \
            (LOG_CHECK_ID % num4 == 0));                                                                              \
  }

#define LOG_CHECK_EVERY_N                                                                        \
  []() {                                                                                         \
    static uint32_t LOG_CHECK_ID = 0;                                                            \
    ++LOG_CHECK_ID;                                                                              \
    return ((LOG_CHECK_ID <= LOG_CHECK_EVERY_FIRSTNUM) ||                                        \
            (LOG_CHECK_ID < LOG_CHECK_EVERY_NUM2 && LOG_CHECK_ID % LOG_CHECK_EVERY_NUM1 == 0) || \
            (LOG_CHECK_ID < LOG_CHECK_EVERY_NUM3 && LOG_CHECK_ID % LOG_CHECK_EVERY_NUM2 == 0) || \
            (LOG_CHECK_ID < LOG_CHECK_EVERY_NUM4 && LOG_CHECK_ID % LOG_CHECK_EVERY_NUM3 == 0) || \
            (LOG_CHECK_ID % LOG_CHECK_EVERY_NUM4 == 0));                                         \
  }

#endif
