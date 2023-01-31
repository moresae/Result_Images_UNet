import os
import cv2
import csv

path = "./roi"
path_save = "./compare"
list_data = []
data = {}

list_img = os.listdir(path)

def compare(frame,filter,name):
    h_filter = frame.shape[0]
    w_filter = frame.shape[1]
    NEW = frame.copy()
    cnt = 0
    for i in range(h_filter):
        for j in range(w_filter):
            c_b = frame[i,j,0]
            c_g = frame[i,j,1]
            c_r = frame[i,j,2]
            if (c_b,c_g,c_r) == filter:
                NEW[i,j,0] = 0
                NEW[i,j,1] = 0
                NEW[i,j,2] = 255
                cnt+=1
            else: 
                NEW[i,j,0] = 0
                NEW[i,j,1] = 0
                NEW[i,j,2] = 0

    percent = round((cnt/(h_filter*w_filter))*100,2)
    cv2.putText(NEW,f"{name}",(10,50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(NEW,f"{cnt}/{h_filter*w_filter}",(10,70),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(NEW,f"{percent}%",(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

    return NEW, cnt

for i in range(0, int(len(list_img)/2)):
    ground = cv2.imread(f"{path}/{i}_ground.png")
    cp_ground = ground.copy()
    ground = cv2.bitwise_not(ground)
    predict = cv2.imread(f"{path}/{i}_predict.png")

    under = cv2.subtract(predict,ground)
    over = cv2.subtract(ground,predict)
    mask = cv2.add(over,under)

    filter = cv2.subtract(ground, mask)
    '''
    VP = Negro [36,0,0]
    FP = Violeta [84,1,68]
    VN = Verde [36,231,121]
    FN = Azul [84,1,0]
    '''
    VP, cnt_vp = compare(filter,(36,0,0),"VP")
    FP, cnt_fp = compare(filter,(84,1,68),"FP")
    VN, cnt_vn = compare(filter,(36,231,121),"VN")
    FN, cnt_fn = compare(filter,(84,1,0),"FN")

    res_compare = cv2.hconcat([VP,FP,VN,FN])
    res_compare = cv2.resize(res_compare,(1300, 700),interpolation = cv2.INTER_AREA)

    h_filter = cv2.hconcat([cp_ground,filter,predict])
    h_filter = cv2.resize(h_filter,(1300, 700),interpolation = cv2.INTER_AREA)

    hi_filter = filter.shape[0]
    wi_filter = filter.shape[1]
    sensitivity = cnt_vp / (cnt_vp + cnt_fn)
    specificity = cnt_vn / (cnt_vn + cnt_fp)
    fpr = cnt_fp / (cnt_vn + cnt_fp)
    fnr = cnt_fn / (cnt_vp + cnt_fn)
    prevalence = (cnt_vp + cnt_fn) / (hi_filter*wi_filter)

    # print(f"Sensitivity: {sensitivity}")
    # print(f"Specificity: {specificity}")
    # print(f"FPR: {fpr}")
    # print(f"FNR: {fnr}")
    # print(f"Prevalence: {prevalence}")

    data['img'] = i
    data['vp'] = cnt_vp
    data['fp'] = cnt_fp
    data['vn'] = cnt_vn
    data['fn'] = cnt_fn
    data['sensitivity'] = sensitivity
    data['specificity'] = specificity
    data['fpr'] = fpr
    data['fnr'] = fnr
    data['prevalence'] = prevalence

    list_data.append(data)
    # print(list_data)

    cv2.putText(res_compare,f"Sensitivity: {sensitivity}",(10, hi_filter-90),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(res_compare,f"Specificity: {specificity}",(10, hi_filter-70),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(res_compare,f"FPR: {fpr}",(10, hi_filter-50),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(res_compare,f"FNR: {fnr}",(10, hi_filter-30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(res_compare,f"Prevalence: {prevalence}",(10, hi_filter-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)

    # cv2.imshow("FILTER", h_filter)
    # cv2.imshow("COMPARE", res_compare)

    cv2.imwrite(f"{path_save}/filter_{i}.png",h_filter)
    cv2.imwrite(f"{path_save}/compare_{i}.png",res_compare)
 
    if cv2.waitKey(0) & 0xFF == ord('ñ'):
        break

data_file = open('out_compare.csv', 'w')
csv_writer = csv.writer(data_file)
count = 0
for dat in list_data:
    if count == 0: 
        header = dat.keys()
        csv_writer.writerow(header)
        count += 1
    csv_writer.writerow(dat.values())
data_file.close()

