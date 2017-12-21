#include <igl/readOFF.h>
#include <igl/viewer/Viewer.h>
#include <igl/readOFF.h>
#include "windows.h"
#include <igl/viewer/Viewer.h>
#include <igl/jet.h>
#include "tutorial_shared_path.h"
#include <igl/unproject_onto_mesh.h>
//#include "sgwt_getparameter.h"
//#include "sgwt_cheby_op.h"
//#include "sgwt_inverse.h"
#include "cstring"
#include "igl/triangle_triangle_adjacency.h"
#include "igl/vertex_triangle_adjacency.h"
#include "iostream"
#include "compute_mesh_laplacian.h"
#include "sgwt_toolbox.h"
#include "GN.h"
#include "time.h"
#include "vector"
using namespace std;

typedef struct ExtremumInLayer
{
	int index;  //from 0;
	double extremum;
};

Eigen::MatrixXi T, TTi;
Eigen::MatrixXd   CV, U;
vector<vector<int>> VF, VI, VV;
int *isColored, *b;
vector<ExtremumInLayer> *extremum_in_layer_x, *extremum_in_layer_y, *extremum_in_layer_z;

int main(int argc, char *argv[])
{

	Eigen::MatrixXi F;
	Eigen::MatrixXd V, Co;
	int nscales = 4, chebyOrder = 50;
	// Load a mesh in OFF format
	igl::readOFF(TUTORIAL_SHARED_PATH "/lion.off", V, F);

	isColored = new int[V.rows() + 1];
	for (int i = 0; i < V.rows(); i++)
		isColored[i] = 0;
	extremum_in_layer_x = new vector<ExtremumInLayer>[nscales+1];
	extremum_in_layer_y = new vector<ExtremumInLayer>[nscales+1];
	extremum_in_layer_z = new vector<ExtremumInLayer>[nscales+1];
	Co = Eigen::MatrixXd::Constant(F.rows(), 3, 1);

	//获取点相邻面和边相邻面
	igl::triangle_triangle_adjacency(F, T, TTi);
	igl::vertex_triangle_adjacency(V.rows(), F, VF, VI);
	

	F.array() += 1;

	Laplacian_Type type = Laplacian_Type::distance;
	Option options(0, 1);
	SpMat L = compute_mesh_laplacian(V, F, type, options);

	Sgwt sgwt(chebyOrder, nscales, L);

	cout << "L is computed" << endl;

	double lmax = sgwt.sgwt_rough_lmax();

	Varargin var;

	sgwt.sgwt_filter_design(lmax, var);

	sgwt.setArange(0, lmax);

	sgwt.sgwt_cheby_coeff<G0>(0, *sgwt.getG0());

	for (int k = 1; k <= sgwt.getNscales(); k++)
	{
		sgwt.sgwt_cheby_coeff<GN>(k, sgwt.getGN()[k - 1]);
	}

	F.array() -= 1;

	igl::viewer::Viewer viewer;

	//Use mouse to pick faces(Colored Yellow)
	viewer.callback_mouse_down =
		[&V, &F, &Co](igl::viewer::Viewer& viewer, int, int)->bool
	{
		int fid;
		Eigen::Vector3f bc;
		// Cast a ray in the view direction starting from the mouse position
		double x = viewer.current_mouse_x;
		double y = viewer.core.viewport(3) - viewer.current_mouse_y;
		if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core.view * viewer.core.model,
			viewer.core.proj, viewer.core.viewport, V, F, fid, bc))
		{
			// paint hit red
			Co.row(fid) << 1, 1, 0;
			int flag = MessageBox(GetForegroundWindow(), "是否选中基于点的一环邻域", " ", 1);
			if (flag == 1)					//选中边一环邻域
			{
				for (int i = 0; i < F.cols(); i++)
				{
					for (int j = 0; j < VF[F(fid, i)].size(); j++)
					{
						int Vertex_Face_Iter = VF[F(fid, i)][j];
						Co.row(Vertex_Face_Iter) << 1, 1, 0;
						isColored[F(Vertex_Face_Iter, 0)] = 1;
						isColored[F(Vertex_Face_Iter, 1)] = 1;
						isColored[F(Vertex_Face_Iter, 2)] = 1;
					}
				}
			}
			flag = MessageBox(GetForegroundWindow(), "是否选中基于边的一环邻域", " ", 1);
			if (flag == 1)
			{
				for (int i = 0; i < 3; i++)
				{
					int Edge_Face_Iter = T(fid, i);
					Co.row(Edge_Face_Iter) << 1, 1, 0;
					isColored[F(Edge_Face_Iter, 0)] = 1;
					isColored[F(Edge_Face_Iter, 1)] = 1;
					isColored[F(Edge_Face_Iter, 2)] = 1;
				}
			}
			for (int i = 0; i < 3; i++) {
				isColored[F(fid,i)] = 1;
			}
			viewer.data.set_colors(Co);
			return true;
		}
		return false;
	};

	//Use keyboard for Spectral Graph Wavelet Transform
	viewer.callback_key_down =
		[&V, &F, &Co, &sgwt](igl::viewer::Viewer& viewer, char key, int)->bool
	{
		switch (key)
		{
		case 'r':
		case 'R':
			U = V;
			break;
		case 'P':
		case 'p':
		case ' ':
		{
			clock_t start, finish;
			double duration;
			F.array() += 1;
			VectorXd tx(V.rows()), ty(V.rows()), tz(V.rows());
			tx << V.col(0);
			ty << V.col(1);
			tz << V.col(2);

			start = clock();
			vector<VectorXd> wpallx = sgwt.sgwt_cheby_op(tx, sgwt.getc());
			vector<VectorXd> wpally = sgwt.sgwt_cheby_op(ty, sgwt.getc());
			vector<VectorXd> wpallz = sgwt.sgwt_cheby_op(tz, sgwt.getc());

			ExtremumInLayer tempx, tempy, tempz;
			for (int scale = 0; scale < sgwt.getNscales()+1; scale++)
			{
				extremum_in_layer_x[scale].clear();
				extremum_in_layer_y[scale].clear();
				extremum_in_layer_z[scale].clear();
				for (int vertexs = 0; vertexs < V.rows(); vertexs++)
				{
					double maxcoeff_x = wpallx[scale](F(VF[vertexs][0], 0)), maxcoeff_y = wpally[scale](F(VF[vertexs][0], 0)), maxcoeff_z = wpallz[scale](F(VF[vertexs][0], 0));
					double mincoeff_x = wpallx[scale](F(VF[vertexs][0], 0)), mincoeff_y = wpally[scale](F(VF[vertexs][0], 0)), mincoeff_z = wpallz[scale](F(VF[vertexs][0], 0));
					for (int k = 1; k < VF[vertexs].size(); k++)
					{
						//cout << F(VF[j][k], 0) - 1 << endl;
						for (int i = 0; i < 3; i++)
						{
							if (wpallx[scale](F(VF[vertexs][k], i)) > maxcoeff_x)
								maxcoeff_x = wpallx[scale](F(VF[vertexs][k], i));
							if (wpally[scale](F(VF[vertexs][k], i)) > maxcoeff_y)
								maxcoeff_y = wpally[scale](F(VF[vertexs][k], i));
							if (wpallz[scale](F(VF[vertexs][k], i)) > maxcoeff_z)
								maxcoeff_z = wpallz[scale](F(VF[vertexs][k], i));

							if (wpallx[scale](F(VF[vertexs][k], i)) < mincoeff_x)
								mincoeff_x = wpallx[scale](F(VF[vertexs][k], i));
							if (wpally[scale](F(VF[vertexs][k], i)) < mincoeff_y)
								mincoeff_y = wpally[scale](F(VF[vertexs][k], i));
							if (wpallz[scale](F(VF[vertexs][k], i)) < mincoeff_z)
								mincoeff_z = wpallz[scale](F(VF[vertexs][k], i));
						}
					}

					tempx.index = vertexs;
					tempx.extremum = wpallx[scale](vertexs);
					if ((wpallx[scale](vertexs) >= maxcoeff_x) || (wpallx[scale](vertexs) <= mincoeff_x))
					{
						extremum_in_layer_x[scale].push_back(tempx);
					}

					tempy.index = vertexs;
					tempy.extremum = wpally[scale](vertexs);
					if ((wpally[scale](vertexs) >= maxcoeff_y) || (wpally[scale](vertexs) <= mincoeff_y))
					{
						extremum_in_layer_y[scale].push_back(tempy);
					}

					tempz.index = vertexs;
					tempz.extremum = wpallz[scale](vertexs);
					if ((wpallz[scale](vertexs) >= maxcoeff_z) || (wpallz[scale](vertexs) <= mincoeff_z))
					{
						extremum_in_layer_z[scale].push_back(tempz);
					}
				}
			}
			finish = clock();
			duration = (double)(finish - start) / CLOCKS_PER_SEC;

			cout << "time for sgwt_cheby_op is "<< duration<< " seconds"<< endl;

			for (int i = 0; i < sgwt.getNscales()+1; i++) {
				cout << "extremum_in_layer_x<<[" << i <<"].size = " << extremum_in_layer_x[i].size() << endl;
				cout << "extremum_in_layer_y<<[" << i << "].size = " << extremum_in_layer_y[i].size() << endl;
				cout << "extremum_in_layer_z<<[" << i << "].size = " << extremum_in_layer_z[i].size() << endl;
			}
			/*for(int i=0;i<extremum_in_layer_x[1].size();i++)
				cout << extremum_in_layer_x[1][i].index <<" "<<extremum_in_layer_x[1][i].extremum << endl;*/

			int ks = 4;
			//对ks + 1后的小波层做变化
			for (int k = ks; k < wpallx.size(); k++) {
				wpallx[k] = wpallx[k] * 20;
				wpally[k] = wpally[k] * 20;
				wpallz[k] = wpallz[k] * 20;
			}
			/*//对1 - ks的小波层和尺度层做变化
			for (int k = 0; k < ks; k++)
				wpallx[k] = wpallx[k] * 1;

			for (int k = ks; k < wpally.size(); k++)
				wpally[k] = wpally[k] * 20;
			//对1 - ks的小波层和尺度层做变化
			for (int k = 0; k < ks; k++)
				wpally[k] = wpally[k] * 1;

			for (int k = ks; k < wpallz.size(); k++)
				wpallz[k] = wpallz[k] * 20;
			//对1 - ks的小波层和尺度层做变化
			for (int k = 0; k < ks; k++)
				wpallz[k] = wpallz[k] * 1;*/

			start = clock();
			VectorXd tx1 = sgwt.sgwt_inverse(wpallx);
			VectorXd ty1 = sgwt.sgwt_inverse(wpally);
			VectorXd tz1 = sgwt.sgwt_inverse(wpallz);
			finish = clock();
			duration = (double)(finish - start) / CLOCKS_PER_SEC;

			cout << "time for sgwt_inverse is " << duration << " seconds" << endl;

			U = V;
			U.col(0) << tx1;
			U.col(1) << ty1;
			U.col(2) << tz1;
		
			cout << "haha" << endl;

			if (key == 'p' || key == 'P')
			{
				for (int i = 0; i < V.rows(); i++)
				{
					if (!isColored[i])
						U.row(i) << V.row(i);
				}
			}
			F.array() -= 1;
			break;
		}
		default:
			return false;
		}
		//CV = Eigen::MatrixXd::Constant(V.rows(), 3, 1);
		//for (int i = 0; i < extremum_in_layer_x[1].size() ; i++)
		//{
		//	CV.row(extremum_in_layer_x[1][i].index-1) << 1, 0, 0;
			//cout << extremum_in_layer_x[4][i].index << endl;
		//}
		// Send new positions, update normals, recenter
		viewer.data.set_vertices(U);
		//viewer.data.set_points(V, CV);
		viewer.core.align_camera_center(U, F);
		return true;
	};

	viewer.data.set_mesh(V, F);
	//viewer.data.set_points(V, CV);
	//Eigen::VectorXd Z = V.col(0);
	viewer.core.show_lines = false;
	viewer.data.set_colors(Co);

	cout << "Press [p] to transfrom on all verticals" << endl;;
	cout << "Press [space] to transfrom on pickeds verticals" << endl;;
	cout << "Press [r] to reset." << endl;;
	return viewer.launch();
}
