# Billiard Robot Control and Simulation System

## 專案說明
這是一套端到端的撞球機器人系統：從球桌影像擷取、YOLO 球體偵測、幾何規劃與物理模擬，到機械手臂通訊與網頁化操作面板都在同一個專案中。程式碼以 Python 為主，搭配 Flask 網頁後台與少量前端資源，能夠在一台電腦上同時完成視覺、規劃、指令傳輸與賽局管理。

**Core features**
- 單機即可完成從拍照→偵測→規劃→傳送機器人指令的全流程。
- 視覺模組支援內參標定（`tools/calibrate_intrinsics.py`）、角點/口袋 homography (`main/vision/yoloball.py`) 與 YOLO 自訂權重（`best.pt`）。
- 幾何求解器（`main/core`）提供直球、單庫反彈等策略，能用 CLI、GUI、或 Web API 呼叫。
- 雙向 TCP 通訊（`main/communicate`）支援 CONNECTED→MOVING→100→座標→DONE 的握手流程。
- Flask WebApp（`webapp/`）結合登入、賽局歷史、即時相機預覽與一鍵打擊工作流程，方便遠端操作。

## 系統架構與資料流
1. **影像擷取**：`main/vision/yoloball.capture_balls()` 使用 OpenCV 取得影像、讀取內參 (`intrinsics.yaml`) 進行去畸變，並根據 `corner.json` 或 `pockets.json` 計算 pixel→cm homography。
2. **球體偵測**：呼叫 YOLO 權重 (`best.pt`) 偵測球心座標，再利用 homography 轉成桌面座標系（cm）。結果以 JSON 儲存於 `captures_json/cords.json`。
3. **幾何規劃**：`main/run_shot.plan_shot_from_json()` 將偵測結果轉換為 `main/core/BilliardSolver` 的輸入，計算母球角度與落點。必要時可呼叫 `main/gui/visualize.py` 進行模擬顯示。
4. **座標校正與封包**：`main/configs/correction.apply_fudge()` 可在轉換至 mm 之前微調邊緣偏差，最後包裝成 `"angle,x_mm,y_mm"` 格式提供給機器人。
5. **通訊層**：`main/main.py` 與 `webapp/robot.perform_handshaked_strike()` 都遵守相同握手協議，負責連線、同步狀態、發送座標並等待 `DONE`。
6. **操作介面**：可直接執行 CLI / `main/main.py`，或啟動 Flask WebApp (支援 MJPEG live view、賽局管理、統計報表) 從瀏覽器控制。

## 功能模組詳解

### 視覺與校正 (`main/vision`)
- `yoloball.py`：主入口，負責倒數拍照、去畸變、載入 `corner.json`/`pockets.json`、YOLO 偵測與儲存 JSON。`capture_balls()` 可輸入不同攝影機索引、寬高以及是否顯示預覽。
- `capture.py`、`arm_capture.py`：提供不同硬體情境下的影像擷取流程。
- `intrinsics.yaml` 與 `handeye_result.yaml`：分別保存鏡頭內參與手眼校正結果。
- `tools/calibrate_intrinsics.py`、`tools/live_cam_*.py`、`tools/grab_calib.py` 等工具腳本協助收集棋盤格影像、標定與測試鏡頭。
- `corner.json` / `pockets.json`：桌面角點或球袋像素座標，供 homography 與 pocket 幾何使用。
- 產出：`captures_json/*.json` 儲存每次偵測的 timestamp、球編號、信心值與 `x_cm/y_cm`。

### 幾何與模擬 (`main/core`, `main/gui`)
- `core/solver_core.py`：`BilliardSolver` 內建球半徑、桌面尺寸 (`main/configs/table.py`) 與 pocket 配置。會先以角度+距離排序嘗試 pocket，若直球受阻則轉為單庫反彈 (`bank-1`)。
- `core/billiard_api.py`：將 solver 結果轉為 `angle_deg`、ghost ball、pocket id 等資訊。`compute_shot()` 是對外 API。
- `run_shot.py`：讀取 YOLO JSON 後決定目標球 (信心最高、指定編號或 `'min'`)，呼叫 `compute_shot()`，失敗資訊則保存於 `get_last_plan_shot_error()`。
- `gui/visualize.py` 與 `gui/simulator.py`：使用 pygame 繪製球桌、ghost ball、庫球路徑，方便調整算法或 demo。
- `configs/correction.py`：針對邊緣偏差提供 `axis` / `radial` 兩種多項式校正。所有對機器人輸出的座標都可在此微調。

### 通訊與自動流程 (`main/main.py`, `main/communicate`)
- `communicate/tcp.py`、`tcp_communicate.py`：封裝 socket 連線、發送與接收字串（自動處理 newline）。`create_connection()` 會依 `main/configs/setting.py` 內預設主機/port 連線，可用 `ROBOT_HOST`/`ROBOT_PORT` 覆寫。
- `main/main.py`：範例循環。流程：等待 `CONNECTED` → `MOVING` → 觸發 `capture_balls()` → 計算座標 → 送出 `"100"` 再送 payload → 等待 `DONE` → 重新連線。
- `run_shot.py` 與 `main/main.py` 皆可被其他程式匯入 (`main.run_shot.plan_shot_from_json`) 供 WebApp 或 CLI 使用。

### Web 操作平台 (`webapp/`)
- Flask Blueprint 架構：`auth` (登入/註冊)、`game` (賽局邏輯)、`models`/`database` (SQLite + SQLAlchemy)。
- `game.html` 提供開始遊戲、輪次控制、即時相機、規劃結果圖 (由 `webapp/render.py` 輸出) 與操作按鈕（打擊/拍照/保持/換邊）。
- `/video_feed` 利用 `webapp/camera.CameraStreamer` 以 MJPEG 提供登入後即時預覽，必要時會停止預覽避免與拍照流程衝突。
- `webapp/robot.py` 集中封裝 `capture_and_plan()`、`perform_handshaked_strike()` 與 `compute_arm_payload()`，同時呼叫 vision/solver/通訊。
- `webapp/game.py` 除了 API 之外也處理統計（平均擊球誤差、勝率）、賽局歷史查詢與 caching。
- `webapp/config.py` 定義攝影機索引、串流大小、是否實際送出指令 (`SEND_TO_ROBOT`)、檔案儲存路徑等設定。
- 靜態檔案：`templates/` HTML、`static/` CSS/JS。主要 UI 包含 game, camera, auth, stats 等頁面。

### CLI 與工具
- `main/cli/shot_cli.py`：命令列介面，可手動指定 JSON 檔與目標球，支援 `--show` 調試。
- `tests/`：收錄單元測試與示例資料，可用於驗證幾何算法與 JSON 介面。
- 其他工具：
  - `tools/cap_test.py`：快速測試攝影機與解析度。
  - `tools/capture_to_json.py`：批次擷取影像並輸出 JSON。
  - `tools/` 內多個腳本協助資料集管理、YOLO 標註轉換等。

### 設定檔與資料
- `main/configs/setting.py`：機器人伺服器 HOST/PORT，支援環境變數覆寫，亦供 WebApp 取得設定。
- `main/configs/pygame_config.py`：GUI 模擬視窗大小、顏色與繪圖參數。
- `main/configs/table.py`：桌面尺寸（cm/m），供 vision 與 solver 共用。
- `CORDS.json`、`captures_json/`：近期偵測結果；`handeye_result.yaml` 則保存手眼校正 matrix。
- `requirements-web.txt`：啟動 WebApp 所需 Python 套件清單；另外 `package.json` 提供前端相關工具。

## 典型操作流程
1. **硬體準備**：固定攝影機、確認光源、連接機器人控制器。若跨網路連線，事先設定 ZeroTier 或 VPN。
2. **鏡頭標定**：使用 `tools/grab_calib.py` 收集棋盤格，接著執行 `python tools/calibrate_intrinsics.py <images> --pattern-cols ... --square-mm ... --out main/vision/intrinsics.yaml` 取得 K/D。
3. **桌面角點／袋口建立**：透過 `tools/homography_helper.py` 或自訂腳本取得 `corner.json` / `pockets.json`（像素座標）。
4. **測試偵測**：`python main/vision/yoloball.py` 或於 WebApp 的打擊流程中觀察 `captures_json/cords.json`，確認球號與信心值。
5. **規劃驗證**：`python main/run_shot.py captures_json/cords.json min --show` 或 `python main/gui/simulator.py` 查看 ghost ball/rail 點。
6. **手臂溝通**：執行 `python main/main.py` 進入自動循環；若要透過 Web 操作，啟動 `python -m webapp.app` 並從瀏覽器呼叫 `/api/strike`。
7. **調整校正**：若邊緣命中偏差，可編輯 `main/configs/correction.py` 係數，再透過 `compute_arm_payload()` 觀察修正結果。

## 安裝與執行

### Python/Robot Pipeline
1. 建議使用 Python 3.8 以上 + `pip` 或 `pipenv` 建立虛擬環境。
2. 安裝必要套件（若僅需 WebApp，可直接使用 `requirements-web.txt`；若執行 YOLO，需自行安裝 `torch`, `ultralytics`, `opencv-python`, `numpy`, `pygame`, `flask`, `sqlalchemy`, `flask-login`, `flask-wtf`, `Pillow` 等）。
3. 取得 YOLO 權重 (`best.pt`) 並放在 `tools/` 或 `main/vision/`，`yoloball.py` 會依 `_pick_model_path()` 自動尋找。
4. 執行 `python main/main.py`（或 `python -m main.main`）啟動握手循環。該腳本會自動：
   - 連到 `ROBOT_HOST:ROBOT_PORT`。
   - 等待 `CONNECTED`/`MOVING`。
   - 呼叫 `capture_balls()`→`plan_shot_from_json()`→`apply_fudge()`。
   - 傳送 `"100"` + 座標並等待 `DONE`。

### CLI / 測試工具
- `python main/run_shot.py captures_json/cords.json 5 --show`：指定球號 5，並用 pygame 顯示規劃。
- `python main/cli/shot_cli.py --help`：查看 CLI 選項，可選擇輸出詳細 debug。
- `pytest`（於 repo root）可執行單元測試（若 `tests/` 提供）。

### WebApp
1. 安裝 Web 依賴：
   ```bash
   pip3 install -r requirements-web.txt
   ```
2. 啟動伺服器（可透過環境變數覆寫 host/port/debug）：
   ```bash
   WEB_HOST=0.0.0.0 WEB_PORT=8000 WEB_DEBUG=true python3 -m webapp.app
   ```
3. 連到 `http://localhost:8000`（或遠端 IP），註冊帳號後即可進入 Game 頁面操作。
4. 若要啟用實機打擊，請在 `webapp/config.py` 設定 `SEND_TO_ROBOT = True`，並確保 `main/configs/setting.py` 中的 HOST/PORT 正確或以環境變數覆蓋。

### Web UI 功能亮點
- **Game Dashboard (`webapp/templates/game.html`)**：建立遊戲、決定難度/先後攻、控制打擊或僅拍照。每個 API (`/api/strike`, `/api/capture_only`, `/api/keep_turn`, `/api/switch_turn`) 皆會更新當前輪次與剩餘球數。
- **Camera 頁面**：透過 `/video_feed` 取得 MJPEG 串流；每次捕捉或打擊前會暫停預覽以避免攝影機資源被佔用。
- **統計/歷史**：`webapp/models.py` 定義 `Game`, `Shot`, `User`；`webapp/game.py` 使用 SQLAlchemy query + cache 計算平均擊球步驟、勝率等。
- **渲染結果**：`webapp/render.py` 會將 YOLO 偵測結果與規劃線條輸出成圖片，於 Game 頁面的「規劃圖」顯示。

### External / ZeroTier Access
可透過區網或 ZeroTier 虛擬網路遠端操作 Web UI 與攝影機：
- 使用 `WEB_HOST=0.0.0.0 WEB_PORT=8000` 對外開放。
- 透過 ZeroTier 互連時，`ROBOT_HOST` / `ROBOT_PORT` 可設定為對方節點的虛擬 IP。
- 連線流程範例：
  ```bash
  WEB_HOST=0.0.0.0 WEB_PORT=8000 WEB_DEBUG=true python3 -m webapp.app
  ROBOT_HOST=10.147.0.23 ROBOT_PORT=4000 python3 -m webapp.app
  ```
- 從其他裝置：`http://<your-host-ip-or-zerotier-ip>:8000/`。

**ZeroTier Checklist**
1. 兩台機器加入同一 ZeroTier 網路並通過授權。
2. 取得伺服器的 ZeroTier IP (`zerotier-cli listpeers` 或網頁控制台)。
3. 以 `0.0.0.0` 及固定 port 啟動 WebApp。
4. 作業系統防火牆需允許該 port 的 TCP 連線。
5. 在客戶端使用 `http://<zerotier-ip>:8000/` 訪問。

**Live Camera / Robot 注意事項**
- `/video_feed` 為 MJPEG 串流且需登入授權。
- 捕捉與實際打擊前會暫停預覽，避免攝影機同時被多個流程開啟。
- 建議在生產部署時加上 Nginx/Caddy 反向代理並啟用 TLS，同時關閉 `WEB_DEBUG`。

### Troubleshooting
- **外部裝置無法連線**：
  - 確認 `ping <zerotier-ip>` 通。
  - 檢查伺服器 `ss -tulpn | rg :8000` 或 `netstat -an | grep 8000` 是否 listening。
  - 防火牆是否允許 `WEB_PORT`。
  - 伺服器上可 `curl http://127.0.0.1:8000/` 測試。
- **實況相機無畫面**：
  - 確定沒有其他程式占用攝影機。
  - 降低解析度於 `webapp/config.py` (`CAMERA_WIDTH/HEIGHT` 或 `CAMERA_STREAM_MAX_WIDTH`)。
  - 執行 `tools/cap_test.py` 驗證攝影機。
  - 如果仍無法，檢查 OpenCV 是否支援該驅動。
- **機器人未收到座標**：
  - 確認 `ROBOT_HOST`/`ROBOT_PORT` 可 `nc -vz <host> <port>`。
  - 檢查 `main/configs/setting.py` 或環境變數是否一致。
  - 查看 `webapp/robot.py` 或 `main/main.py` 日誌中是否拋出 `plan_shot` 錯誤 (`get_last_plan_shot_error()`)。
- **YOLO 偵測錯誤或球號對不上**：
  - 更新 `MANUAL_CLASS_LIST` 或 `MANUAL_CLASS_MAP` (`main/vision/yoloball.py`) 以強制 mapping。
  - 調整 `CONF_THRES`/`IOU_THRES`。
  - `captures_json/` 內的 JSON 可協助比對實況圖像與偵測結果。
- **邊緣命中偏差**：
  - 在 `main/configs/correction.py` 切換 `FUDGE_MODE` 或調整 KX/KY/K1/K2。
  - `compute_arm_payload()` 會自動套用 `apply_fudge()`，可直接列印比對修改成效。

## 設定與環境變數
- `ROBOT_HOST` / `ROBOT_PORT`：覆寫 `main/configs/setting.py`，供 CLI 與 WebApp 使用。
- `WEB_HOST` / `WEB_PORT` / `WEB_DEBUG`：啟動 Flask 伺服器時指定綁定介面、port 與除錯模式。
- `SEND_TO_ROBOT`（`webapp/config.py`）：是否真的傳送 TCP 指令；預設 False 避免在測試時動到機器手臂。
- `CAMERA_INDEX` / `CAMERA_WIDTH` / `CAMERA_HEIGHT` / `CAMERA_STREAM_MAX_WIDTH`：控制 WebApp 中 `CameraStreamer` 的攝影機來源與解析度。
- `INTRINSICS_PATH`、`corner.json`/`pockets.json` 路徑可在部署時複製到 `main/vision/` 或傳入 `capture_balls` 參數。

## Contributing
歡迎透過 Fork + Pull Request 的方式貢獻：
1. Fork 專案並建立新 branch (`feature/...`、`fix/...`)。
2. 完成修改後撰寫說明充分的 commit message。
3. 推送到遠端並發 PR，描述問題、解法與測試結果。若涉及視覺/機器人實驗，建議附上對應資料或截圖。

## License
若 repo 中提供 `LICENSE`/`LICENSE.md`，請依其條款使用；否則此專案屬專有程式碼，請與作者聯繫。
