import csv
import os
import time
from datetime import datetime

import pandas as pd 
from google import genai
from google.genai import errors as genai_errors


API_KEYS = [
    
]

MODELS = [
    "gemini-2.5-flash",
]

MAX_RETRIES = 5
THROTTLE_SECONDS = 1.0
RATE_LIMIT_PAUSE_SECONDS = 60.0  
API_KEY_SWITCH_DELAY = 60.0 
MODEL_SWITCH_DELAY = 60.0  
PROMPT_TEMPLATE = """Bạn là một chuyên gia gán nhãn ý định của người dùng trong hệ thống thương mại điện tử.

                Hãy đọc câu dưới đây và xác định:
- Nếu người dùng bày tỏ cảm xúc, đánh giá, chê, khen, bình luận, không có ý định hỏi thông tin → label là "chat".
Ví dụ: "iphone 16 pro mau hết pin chỉ lướt Facebook thôi mà 1 phần trăm pin chỉ được có khoảng 4 phút mấy chưa được 5 phút nữa trong khi mới mua 27/10, không đáng mua", chat
- Nếu người dùng hỏi về sản phẩm, giá, tình trạng, chính sách, so sánh ... thì label là "retrieval-lĩnh vực được nhắc đến , nếu lĩnh vục được nhắc đến không phải là điện thoại , tivi, tủ lạnh, máy lạnh , laptop thì trả về mỗi "retrivel"".

Trả về văn bản là một trong hai giá trị: "chat" hoặc "retrieval-lĩnh vực được nhắc đến , nếu lĩnh vục được nhắc đến không phải là điện thoại , tivi, tủ lạnh, máy lạnh , laptop thì trả về mỗi "retrivel"
input:{content}
"""

CSV_PATH = "/home/big/fine-tune-intent/scrapy/all_comments_content_only.csv"
OUTPUT_PATH = "/home/big/fine-tune-intent/scrapy/all_comments_labeled.csv"


def _log(message: str, level: str = "INFO") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def build_prompt(content: str) -> str:
    
    return PROMPT_TEMPLATE.format(content=content.strip())


class RateLimitExceeded(Exception):

    def __init__(
        self,
        message: str,
        retry_after_seconds: float | None = None,
        kind: str = "rate_limit",
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
        self.kind = kind


class APIKeyManager:

    def __init__(self, api_keys: list[str]) -> None:
        if not api_keys:
            raise ValueError("Cần ít nhất một API key")
        self.api_keys = api_keys
        self.current_index = 0
        self.start_index = 0  
        self.client = genai.Client(api_key=self.api_keys[0])

    def get_client(self) -> genai.Client:
        """Trả về client hiện tại."""
        return self.client

    def switch_to_next_key(self) -> bool:
        """Chuyển sang API key tiếp theo. Trả về True nếu chuyển được, False nếu đã quay vòng hết tất cả keys."""
        if len(self.api_keys) == 1:
            
            return False
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        self.client = genai.Client(api_key=self.api_keys[self.current_index])
        
        if self.current_index == self.start_index:
            return False
        return True

    def reset_cycle(self) -> None:
        """Reset vòng quay, đánh dấu key hiện tại là điểm bắt đầu mới."""
        self.start_index = self.current_index

    def get_current_key_index(self) -> int:
        """Trả về index của API key hiện tại."""
        return self.current_index


class ModelManager:
    """Quản lý việc luân phiên giữa các model khi bị quá tải."""

    def __init__(self, models: list[str]) -> None:
        if not models:
            raise ValueError("Cần ít nhất một model")
        self.models = models
        self.current_index = 0
        self.start_index = 0

    def get_model(self) -> str:
        """Trả về model hiện tại."""
        return self.models[self.current_index]

    def switch_to_next_model(self) -> bool:
        """Chuyển sang model tiếp theo. False nếu đã quay vòng hết."""
        if len(self.models) == 1:
            return False
        self.current_index = (self.current_index + 1) % len(self.models)
        if self.current_index == self.start_index:
            return False
        return True

    def reset_cycle(self) -> None:
        """Đánh dấu model hiện tại là điểm bắt đầu cho vòng quay mới."""
        self.start_index = self.current_index

    def get_current_model_index(self) -> int:
        """Trả về index model hiện tại."""
        return self.current_index


def _extract_retry_delay(exception: Exception, default: float = 30.0) -> float:
    """Lấy thời gian chờ gợi ý từ lỗi API nếu có."""
    response_json = getattr(exception, "response_json", {}) or {}
    details = response_json.get("error", {}).get("details", [])
    for detail in details:
        if detail.get("@type", "").endswith("RetryInfo"):
            retry_delay = detail.get("retryDelay")
            if isinstance(retry_delay, str) and retry_delay.endswith("s"):
                try:
                    return float(retry_delay[:-1]) or default
                except ValueError:
                    pass
    return default


def label_text(content: str, client: genai.Client, model_name: str) -> str:
    """Gửi nội dung đến mô hình để lấy nhãn."""
    if not content:
        return ""
    prompt = build_prompt(content)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt > 1:
                _log(f"Thử lại lần {attempt}/{MAX_RETRIES} với model {model_name}", "RETRY")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            label = (response.text or "").strip()
            if attempt > 1:
                _log(f"Thành công sau {attempt} lần thử. Label: {label[:50]}...", "SUCCESS")
            return label
        except genai_errors.ClientError as exc:
            code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            is_429 = (code == 429) or ("429" in str(exc))
            if is_429 and attempt < MAX_RETRIES:
                delay = _extract_retry_delay(exc, default=30.0)
                time.sleep(delay)
                continue
            if is_429:
                # Hết quota trong phiên này -> raise để đổi API key
                raise RateLimitExceeded(
                    "Rate limit/quota exceeded",
                    retry_after_seconds=API_KEY_SWITCH_DELAY,
                    kind="api_key",
                )
            raise
        except genai_errors.ServerError as exc:
            code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            is_503 = (code == 503) or ("503" in str(exc))
            if is_503 and attempt < MAX_RETRIES:
                time.sleep(THROTTLE_SECONDS * (2 ** (attempt - 1)))
                continue
            if is_503:
                # Model quá tải: tạm dừng dài rồi thử lại cùng dòng
                raise RateLimitExceeded(
                    "Model overloaded (503). Pausing before retry.",
                    retry_after_seconds=MODEL_SWITCH_DELAY,
                    kind="model",
                )
            raise
        except Exception:
            if attempt < MAX_RETRIES:
                time.sleep(THROTTLE_SECONDS * (2 ** (attempt - 1)))
                continue
            raise
    return ""


def _load_labeled_row_ids(output_path: str) -> set[int]:
    """Đọc file output hiện có và trả về tập row_id đã gán nhãn (để skip)."""
    if not os.path.exists(output_path):
        return set()
    labeled_ids: set[int] = set()
    with open(output_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        has_row_id = "row_id" in (reader.fieldnames or [])
        if not has_row_id:
            # Không có row_id: không thể mạnh mẽ bỏ qua trùng lặp theo id
            return set()
        for row in reader:
            try:
                labeled_ids.add(int(row["row_id"]))
            except Exception:
                continue
    return labeled_ids


def label_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    "Xử lý lần lượt từng content, dừng 30 phút khi bị limit và tiếp tục đến khi hết dữ liệu. không để  chương trình bị dừng khi chưa hết content được lable"
    _log("=" * 80, "INFO")
    _log("BẮT ĐẦU QUÁ TRÌNH GÁN NHÃN", "INFO")
    _log("=" * 80, "INFO")
    _log(f"File input: {csv_path}", "INFO")
    _log(f"File output: {OUTPUT_PATH}", "INFO")
    _log(f"Số API keys: {len(API_KEYS)}", "INFO")
    _log(f"Số models: {len(MODELS)}", "INFO")
    
    start_time = time.time()
    df = pd.read_csv(csv_path)
    if "content" not in df.columns:
        raise ValueError("CSV không chứa cột 'content'.")

    # Thêm row_id dựa trên index để có thể resume ổn định
    df = df.reset_index().rename(columns={"index": "row_id"})
    total_rows = len(df)
    _log(f"Tổng số dòng trong file: {total_rows}", "INFO")

    # Chuẩn bị file output (append để resume)
    file_exists = os.path.exists(OUTPUT_PATH)
    labeled_ids = _load_labeled_row_ids(OUTPUT_PATH)
    _log(f"Đã có {len(labeled_ids)} dòng đã được gán nhãn trước đó", "INFO")

    # Nếu file output tồn tại nhưng không có row_id, fall-back theo content
    existing_contents: set[str] = set()
    if file_exists and not labeled_ids:
        with open(OUTPUT_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "content" in reader.fieldnames:
                for row in reader:
                    content_val = (row.get("content") or "").strip()
                    if content_val:
                        existing_contents.add(content_val)

    out_fields = ["row_id", "content", "label"]
    # Viết header nếu file chưa tồn tại
    if not file_exists:
        with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=out_fields)
            writer.writeheader()

    # Danh sách các dòng cần xử lý (đã lọc bỏ những dòng có sẵn)
    pending_rows: list[tuple[int, str]] = []
    for _, row in df.iterrows():
        row_id = int(row["row_id"])
        content_val = (row["content"] or "").strip()
        if labeled_ids and row_id in labeled_ids:
            continue
        if not labeled_ids and existing_contents and content_val in existing_contents:
            continue
        pending_rows.append((row_id, content_val))

    if not pending_rows:
        _log("Tất cả dòng đã được gán nhãn. Không cần xử lý thêm.", "INFO")
        return pd.read_csv(OUTPUT_PATH) if os.path.exists(OUTPUT_PATH) else df.assign(label="")

    # Khởi tạo API key/model manager
    key_manager = APIKeyManager(API_KEYS)
    model_manager = ModelManager(MODELS)
    _log(f"Cần xử lý {len(pending_rows)} dòng", "INFO")
    _log(f"Bắt đầu với API key {key_manager.get_current_key_index() + 1}/{len(API_KEYS)}", "INFO")
    _log(f"Bắt đầu với model {model_manager.get_model()} ({model_manager.get_current_model_index() + 1}/{len(MODELS)})", "INFO")
    _log("-" * 80, "INFO")

    with open(OUTPUT_PATH, "a", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=out_fields)
        idx = 0
        consecutive_api_limit_count = 0  # Đếm số lần 429 liên tiếp với cùng một key
        consecutive_model_limit_count = 0  # Đếm số lần 503 liên tiếp với cùng một model
        last_progress_log = time.time()

        while idx < len(pending_rows):
            row_id, content_val = pending_rows[idx]
            progress = (idx / len(pending_rows)) * 100
            elapsed = time.time() - start_time
            
            # Log tiến độ mỗi 10 dòng hoặc mỗi 30 giây
            if idx % 10 == 0 or (time.time() - last_progress_log) >= 30:
                remaining = len(pending_rows) - idx
                avg_time_per_row = elapsed / (idx + 1) if idx > 0 else 0
                estimated_remaining = avg_time_per_row * remaining
                _log(f"Tiến độ: {idx}/{len(pending_rows)} ({progress:.1f}%) | "
                     f"Đã xử lý: {idx} | Còn lại: {remaining} | "
                     f"Thời gian: {elapsed/60:.1f} phút | "
                     f"Ước tính còn: {estimated_remaining/60:.1f} phút", "PROGRESS")
                last_progress_log = time.time()
            
            try:
                client = key_manager.get_client()
                model_name = model_manager.get_model()
                current_key_idx = key_manager.get_current_key_index() + 1
                current_model_idx = model_manager.get_current_model_index() + 1
                
                _log(f"[{idx+1}/{len(pending_rows)}] Row ID: {row_id} | "
                     f"API Key: {current_key_idx}/{len(API_KEYS)} | "
                     f"Model: {model_name} ({current_model_idx}/{len(MODELS)})", "PROCESSING")
                _log(f"Content: {content_val[:100]}{'...' if len(content_val) > 100 else ''}", "DETAIL")
                
                request_start = time.time()
                label = label_text(content_val, client, model_name)
                request_time = time.time() - request_start
                
                writer.writerow({"row_id": row_id, "content": content_val, "label": label})
                f_out.flush()
                os.fsync(f_out.fileno())
                idx += 1
                consecutive_api_limit_count = 0
                consecutive_model_limit_count = 0
                
                _log(f"✓ Thành công! Label: {label[:80]}{'...' if len(label) > 80 else ''} | "
                     f"Thời gian: {request_time:.2f}s", "SUCCESS")
                time.sleep(THROTTLE_SECONDS)
            except RateLimitExceeded as exc:
                delay = exc.retry_after_seconds or 1.0
                reason = getattr(exc, "kind", "rate_limit")

                if reason == "api_key":
                    consecutive_api_limit_count += 1
                    consecutive_model_limit_count = 0
                    current_key_idx = key_manager.get_current_key_index() + 1

                    _log(f"Rate limit (429) với API key {current_key_idx}/{len(API_KEYS)} | "
                         f"Lần thứ {consecutive_api_limit_count}", "WARNING")

                    # Nếu đã thử với key hiện tại và vẫn bị 429, đổi sang key khác
                    if consecutive_api_limit_count >= 2:
                        _log(f"API key {current_key_idx} bị limit 2 lần liên tiếp. "
                             f"Đang chuyển sang key khác...", "WARNING")
                        switched = key_manager.switch_to_next_key()
                        if not switched:
                            # Đã quay vòng hết tất cả keys, đợi lâu hơn
                            _log(f"Tất cả {len(API_KEYS)} API keys đều bị limit. "
                                 f"Đợi {delay} giây trước khi thử lại từ đầu...", "WARNING")
                            time.sleep(delay)
                            consecutive_api_limit_count = 0
                            key_manager.reset_cycle()
                            _log("Đã reset vòng quay API keys", "INFO")
                        else:
                            new_key_idx = key_manager.get_current_key_index() + 1
                            _log(f"✓ Đã chuyển sang API key {new_key_idx}/{len(API_KEYS)}", "SUCCESS")
                            consecutive_api_limit_count = 0
                    else:
                        # Lần đầu gặp 429 với key này, đợi 1 phút rồi thử lại
                        _log(f"Đợi {delay} giây trước khi thử lại với API key {current_key_idx}...", "INFO")
                        time.sleep(delay)
                elif reason == "model":
                    consecutive_model_limit_count += 1
                    consecutive_api_limit_count = 0
                    current_model = model_manager.get_model()
                    current_model_idx = model_manager.get_current_model_index() + 1

                    _log(f"Model quá tải (503) với model {current_model} "
                         f"({current_model_idx}/{len(MODELS)}) | Lần thứ {consecutive_model_limit_count}", "WARNING")

                    if consecutive_model_limit_count >= 2:
                        _log(f"Model {current_model} bị quá tải 2 lần liên tiếp. "
                             f"Đang chuyển sang model khác...", "WARNING")
                        switched_model = model_manager.switch_to_next_model()
                        if not switched_model:
                            _log(f"Tất cả {len(MODELS)} models đều bị quá tải. "
                                 f"Đợi {delay} giây trước khi thử lại từ đầu...", "WARNING")
                            time.sleep(delay)
                            consecutive_model_limit_count = 0
                            model_manager.reset_cycle()
                            _log("Đã reset vòng quay models", "INFO")
                        else:
                            new_model = model_manager.get_model()
                            new_model_idx = model_manager.get_current_model_index() + 1
                            _log(f"✓ Đã chuyển sang model {new_model} ({new_model_idx}/{len(MODELS)})", "SUCCESS")
                            consecutive_model_limit_count = 0
                    else:
                        _log(f"Đợi {delay} giây trước khi thử lại với model {current_model}...", "INFO")
                        time.sleep(delay)
                else:
                    consecutive_api_limit_count = 0
                    consecutive_model_limit_count = 0
                    _log(f"Lỗi tạm thời: {exc}. Đợi {delay} giây trước khi thử lại...", "WARNING")
                    time.sleep(delay)
                # không tăng idx để thử lại cùng dòng sau khi đợi/đổi key
            except Exception as e:
                _log(f"LỖI KHÔNG XỬ LÝ ĐƯỢC: {type(e).__name__}: {str(e)}", "ERROR")
                _log(f"Row ID: {row_id} | Content: {content_val[:100]}...", "ERROR")
                raise

    # Đọc lại toàn bộ output làm kết quả trả về
    total_time = time.time() - start_time
    result_df = pd.read_csv(OUTPUT_PATH)
    _log("-" * 80, "INFO")
    _log("=" * 80, "INFO")
    _log("HOÀN THÀNH QUÁ TRÌNH GÁN NHÃN", "INFO")
    _log("=" * 80, "INFO")
    _log(f"Tổng số dòng đã xử lý: {len(result_df)}", "INFO")
    _log(f"Tổng thời gian: {total_time/60:.2f} phút ({total_time:.2f} giây)", "INFO")
    if len(pending_rows) > 0:
        avg_time = total_time / len(pending_rows)
        _log(f"Thời gian trung bình mỗi dòng: {avg_time:.2f} giây", "INFO")
    _log(f"File output: {OUTPUT_PATH}", "INFO")
    _log("=" * 80, "INFO")
    return result_df


def main() -> None:
    """Chạy quá trình gán nhãn trên file CSV đầu vào và lưu kết quả."""
    try:
        df_labeled = label_data(CSV_PATH)
        df_labeled.to_csv(OUTPUT_PATH, index=False)
        _log("Đã lưu kết quả vào file output", "INFO")
    except KeyboardInterrupt:
        _log("Chương trình bị dừng bởi người dùng (Ctrl+C)", "WARNING")
        _log("Dữ liệu đã xử lý đã được lưu vào file output", "INFO")
        raise
    except Exception as e:
        _log(f"Chương trình gặp lỗi và dừng: {type(e).__name__}: {str(e)}", "ERROR")
        _log("Dữ liệu đã xử lý trước đó đã được lưu vào file output", "INFO")
        raise


if __name__ == "__main__":
    main()