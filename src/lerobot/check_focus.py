import cv2

# 打开手眼相机 (ID: 2)
cap = cv2.VideoCapture(0)

print("按 'q' 键退出预览。")
while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取画面，请检查相机是否被占用！")
        break
    
    # 显示实时画面
    cv2.imshow("Hand Camera Preview - Focus Check", frame)
    
    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()