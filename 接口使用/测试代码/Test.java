package com.AiString.test;


import java.io.IOException;

import com.xiaoxian.api.Func;

import net.sf.json.JSONArray;

public class Test {
	public static void main(String[] args) {
		try {
			byte[] data1 = Func.getImageData("pyfile/pic/people/1.jpg");
			byte[] data2 = Func.getImageData("pyfile/pic/room/1.jpg");
			JSONArray result = Func.func(data1,data2, 0, 0, 0.0, 120, 1);
			System.out.println(result.toString());
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
